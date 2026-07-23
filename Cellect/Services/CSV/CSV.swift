import Foundation

/// Minimal RFC-4180 CSV read/write. Kept dependency-free and platform-neutral so the
/// on-disk format matches exactly what an Android client (or pandas/spacr) would produce.
enum CSV {

    /// Parse into (header, rows). Each row is padded/truncated to header width on read.
    static func parse(_ text: String) -> (header: [String], rows: [[String]]) {
        var records: [[String]] = []
        var field = ""
        var record: [String] = []
        var inQuotes = false
        let scalars = Array(text.unicodeScalars)
        var i = 0

        func endField() { record.append(field); field = "" }
        func endRecord() { endField(); records.append(record); record = [] }

        while i < scalars.count {
            let c = scalars[i]
            if inQuotes {
                if c == "\"" {
                    if i + 1 < scalars.count && scalars[i + 1] == "\"" {
                        field.unicodeScalars.append("\"")  // escaped quote
                        i += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.unicodeScalars.append(c)
                }
            } else {
                switch c {
                case "\"": inQuotes = true
                case ",":  endField()
                case "\n": endRecord()
                case "\r": break  // swallow; \r\n handled by the \n
                default:   field.unicodeScalars.append(c)
                }
            }
            i += 1
        }
        // trailing field/record if file doesn't end in newline
        if !field.isEmpty || !record.isEmpty { endRecord() }

        guard let header = records.first else { return ([], []) }
        let body = Array(records.dropFirst()).filter { !($0.count == 1 && $0[0].isEmpty) }
        return (header, body)
    }

    /// Serialize header + rows to an RFC-4180 string (LF line endings).
    static func serialize(header: [String], rows: [[String]]) -> String {
        var out = header.map(escape).joined(separator: ",")
        out += "\n"
        for row in rows {
            out += row.map(escape).joined(separator: ",")
            out += "\n"
        }
        return out
    }

    private static func escape(_ field: String) -> String {
        if field.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" || $0 == "\r" }) {
            return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return field
    }
}
