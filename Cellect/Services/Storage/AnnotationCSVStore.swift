import Foundation

/// Reads and writes the folder's annotation CSV while preserving any columns Cellect
/// didn't create (per the CSV contract in docs/ARCHITECTURE.md §4). `filename` is the join key.
struct AnnotationCSVStore {
    let provider: StorageProvider
    let csvFileName: String

    /// Load existing records keyed by filename. Empty if the CSV doesn't exist yet.
    func load() async throws -> (records: [String: AnnotationRecord], externalColumns: [String]) {
        guard let data = try await provider.readFile(named: csvFileName),
              let text = String(data: data, encoding: .utf8) else {
            return ([:], [])
        }
        let (header, rows) = CSV.parse(text)
        guard let fileIdx = header.firstIndex(of: "filename") else { return ([:], []) }

        // Columns other than "filename" — preserved verbatim on write-back.
        let externalColumns = header.enumerated()
            .filter { $0.offset != fileIdx }
            .map { $0.element }

        var records: [String: AnnotationRecord] = [:]
        for row in rows where row.indices.contains(fileIdx) {
            let filename = row[fileIdx]
            guard !filename.isEmpty else { continue }
            var values: [String: String] = [:]
            for (idx, col) in header.enumerated() where idx != fileIdx {
                values[col] = row.indices.contains(idx) ? row[idx] : ""
            }
            records[filename] = AnnotationRecord(filename: filename, values: values)
        }
        return (records, externalColumns)
    }

    /// Merge our records over whatever is on disk and write the CSV atomically.
    /// `orderedFilenames` fixes row order (folder listing order); `columnOrder` fixes
    /// the header order of Cellect-managed columns. Unknown on-disk columns are appended.
    func save(
        records: [String: AnnotationRecord],
        orderedFilenames: [String],
        columnOrder: [String]
    ) async throws {
        let (existing, externalColumns) = try await load()

        // Merge: start from disk, overlay in-memory values.
        var merged = existing
        for name in orderedFilenames {
            var rec = merged[name] ?? AnnotationRecord(filename: name)
            if let ours = records[name] {
                for (k, v) in ours.values { rec.values[k] = v }
            }
            merged[name] = rec
        }

        // Header: filename, our columns (in order), then any external columns we didn't manage.
        let managed = columnOrder
        let extras = externalColumns.filter { !managed.contains($0) }
        let header = ["filename"] + managed + extras

        // Row order: folder order first, then any filenames only present on disk.
        var seen = Set(orderedFilenames)
        var rowNames = orderedFilenames
        for name in merged.keys where !seen.contains(name) {
            rowNames.append(name); seen.insert(name)
        }

        let rows: [[String]] = rowNames.map { name in
            let rec = merged[name]
            return header.map { col in
                if col == "filename" { return name }
                return rec?.values[col] ?? ""
            }
        }

        let text = CSV.serialize(header: header, rows: rows)
        let data = Data(text.utf8)
        try await provider.writeFile(named: csvFileName, data: data)
    }
}
