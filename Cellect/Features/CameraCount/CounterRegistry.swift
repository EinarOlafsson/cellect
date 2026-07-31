import CoreML
import Foundation

/// Quality/heaviness tiers. Higher = more accurate + more compute. `classical` is the floor.
enum CounterTier: Int, Comparable, CaseIterable, Codable {
    case classical, low, medium, high, best
    static func < (a: CounterTier, b: CounterTier) -> Bool { a.rawValue < b.rawValue }

    var title: String {
        switch self {
        case .classical: return "Classical CV"
        case .low:       return "Light model"
        case .medium:    return "Balanced model"
        case .high:      return "Accurate model"
        case .best:      return "Largest model"
        }
    }
}

/// Rough device-capability recommendation. It never hides a model from the picker.
enum DeviceCapability {
    static var physicalMemoryGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
    }

    static var tierCeiling: CounterTier {
        let gb = physicalMemoryGB
        if gb >= 7.5 { return .best }
        if gb >= 5.5 { return .high }
        if gb >= 3.5 { return .medium }
        return .low
    }
}

/// Describes one workstation-trained foreground/boundary segmentation network.
struct CoreMLCounterSpec: Identifiable {
    let id: String
    let tier: CounterTier
    let modelResourceName: String
    let displayName: String
    let architecture: String
    let inputSize: Int
    let parameterCount: Int
    let testDice: Double
    let testIoU: Double
    let countMAE: Double
    let precision: String
    /// SHA-256 of the verified TorchScript artifact used for this Core ML export.
    let sourceTorchScriptSHA256: String
    let isRecommended: Bool

    var exportVersion: String { "1" }

    /// All nine models returned by the full RTX 3090 workstation run.
    /// Order is held-out Dice, best first, so comparisons are easy in the app.
    static let bundled: [CoreMLCounterSpec] = [
        CoreMLCounterSpec(
            id: "efficientnet_b3_unetpp",
            tier: .best,
            modelResourceName: "CellectEfficientNetB3",
            displayName: "EfficientNet-B3 U-Net++",
            architecture: "U-Net++",
            inputSize: 512,
            parameterCount: 13_624_938,
            testDice: 0.919326,
            testIoU: 0.854746,
            countMAE: 189.10,
            precision: "Float16",
            sourceTorchScriptSHA256: "86c0629091f652709bf45f7622e96068e2d64ef069537e42dd72ee2e6ea51ac4",
            isRecommended: true
        ),
        CoreMLCounterSpec(
            id: "segformer_b2",
            tier: .high,
            modelResourceName: "CellectSegFormerB2",
            displayName: "SegFormer-B2",
            architecture: "SegFormer",
            inputSize: 512,
            parameterCount: 24_722_626,
            testDice: 0.917482,
            testIoU: 0.851773,
            countMAE: 242.45,
            precision: "Float32",
            sourceTorchScriptSHA256: "dc65b0a81d8a1e35bffc53040739d826b93297fc276c51d440dc2f550813967b",
            isRecommended: false
        ),
        CoreMLCounterSpec(
            id: "segformer_b5",
            tier: .best,
            modelResourceName: "CellectSegFormerB5",
            displayName: "SegFormer-B5",
            architecture: "SegFormer",
            inputSize: 512,
            parameterCount: 81_969_346,
            testDice: 0.917453,
            testIoU: 0.851725,
            countMAE: 248.60,
            precision: "Float32",
            sourceTorchScriptSHA256: "584618ed9ed242d5c88ea445fae275b3b6a32a65c7e4e2f3e06d38eb27910853",
            isRecommended: false
        ),
        CoreMLCounterSpec(
            id: "efficientnet_b0_unet",
            tier: .medium,
            modelResourceName: "CellectEfficientNetB0",
            displayName: "EfficientNet-B0 U-Net",
            architecture: "U-Net",
            inputSize: 512,
            parameterCount: 6_251_614,
            testDice: 0.916617,
            testIoU: 0.850175,
            countMAE: 203.84,
            precision: "Float16",
            sourceTorchScriptSHA256: "71c562ab5dfa79bb97e3d7b3a4d1f67bc1ec16e4659e7411c4f5d26c2ee4d5a1",
            isRecommended: false
        ),
        CoreMLCounterSpec(
            id: "resnet101_unetpp",
            tier: .best,
            modelResourceName: "CellectResNet101",
            displayName: "ResNet101 U-Net++",
            architecture: "U-Net++",
            inputSize: 512,
            parameterCount: 67_978_018,
            testDice: 0.916189,
            testIoU: 0.849885,
            countMAE: 194.74,
            precision: "Float32",
            sourceTorchScriptSHA256: "32b68d7456ace782c27b9d6c77c576dd0c4273bdba585f12201445f82a563a50",
            isRecommended: false
        ),
        CoreMLCounterSpec(
            id: "resnet18_unet",
            tier: .medium,
            modelResourceName: "CellectResNet18",
            displayName: "ResNet18 U-Net",
            architecture: "U-Net",
            inputSize: 512,
            parameterCount: 14_328_354,
            testDice: 0.914861,
            testIoU: 0.847359,
            countMAE: 207.37,
            precision: "Float16",
            sourceTorchScriptSHA256: "c7dd6f1f175c2cc3f1d7e592d60e5512d99de83cc59fbf7e8c11701e75efea64",
            isRecommended: false
        ),
        CoreMLCounterSpec(
            id: "resnet50_deeplab",
            tier: .high,
            modelResourceName: "CellectResNet50DeepLab",
            displayName: "ResNet50 DeepLabV3+",
            architecture: "DeepLabV3+",
            inputSize: 512,
            parameterCount: 26_677_842,
            testDice: 0.913699,
            testIoU: 0.845642,
            countMAE: 250.36,
            precision: "Float32",
            sourceTorchScriptSHA256: "9fd2de6c36d4ef26b64e1f105715e883740aab7a0e8b2d62d2495b525358970c",
            isRecommended: false
        ),
        CoreMLCounterSpec(
            id: "mobilenetv3_large_deeplab",
            tier: .low,
            modelResourceName: "CellectMobileNetV3Large",
            displayName: "MobileNetV3-Large DeepLabV3+",
            architecture: "DeepLabV3+",
            inputSize: 512,
            parameterCount: 4_708_610,
            testDice: 0.907898,
            testIoU: 0.835782,
            countMAE: 260.03,
            precision: "Float32",
            sourceTorchScriptSHA256: "3fbd296274211fdf6558cd81820d92426c27c84ff02a6d934fd9031712baf7a1",
            isRecommended: false
        ),
        CoreMLCounterSpec(
            id: "mobilenetv3_small_unet",
            tier: .low,
            modelResourceName: "CellectMobileNetV3Small",
            displayName: "MobileNetV3-Small U-Net",
            architecture: "U-Net",
            inputSize: 384,
            parameterCount: 3_585_794,
            testDice: 0.880045,
            testIoU: 0.792299,
            countMAE: 256.90,
            precision: "Float32",
            sourceTorchScriptSHA256: "655f60805c3efa069a91dc7ead082ef4c28b64a46cea3f3b86b5286f261b49d9",
            isRecommended: false
        ),
    ]

    var isAvailable: Bool {
        Bundle.main.url(forResource: modelResourceName, withExtension: "mlmodelc") != nil
    }
}

/// Exposes every compiled model. Hardware recommendations never remove choices.
enum CounterRegistry {
    /// Workstation-trained models that are compiled into this build, in evaluation order.
    static func availableModelSpecs() -> [CoreMLCounterSpec] {
        CoreMLCounterSpec.bundled.filter(\.isAvailable)
    }

    static func availableCounters() -> [CellCounter] {
        let trained: [CellCounter] = availableModelSpecs()
            .map { CoreMLCellCounter(spec: $0) }
        return trained + [ClassicalCellCounter()]
    }

    static func counter(for identifier: String) -> CellCounter? {
        availableCounters().first { $0.identifier == identifier }
    }

    static func spec(for identifier: String) -> CoreMLCounterSpec? {
        CoreMLCounterSpec.bundled.first { $0.id == identifier && $0.isAvailable }
    }

    /// Resolve a user-selected ensemble without allowing duplicate or unavailable members.
    static func specs(for identifiers: [String]) -> [CoreMLCounterSpec] {
        var seen = Set<String>()
        return identifiers.compactMap { identifier in
            guard seen.insert(identifier).inserted else { return nil }
            return spec(for: identifier)
        }
    }

    static func ensembleCounter(
        identifiers: [String],
        weights: [String: Double],
        strategy: EnsembleMergeStrategy
    ) -> CoreMLEnsembleCellCounter? {
        let selected = specs(for: identifiers)
        let uniqueIdentifierCount = Set(identifiers).count
        guard identifiers.count == uniqueIdentifierCount,
              selected.count == uniqueIdentifierCount,
              selected.count >= 2 else { return nil }
        if strategy.usesModelWeights {
            let selectedWeights = identifiers.compactMap { weights[$0] }
            guard selectedWeights.count == identifiers.count,
                  selectedWeights.allSatisfy({ $0.isFinite && $0 >= 0 }),
                  selectedWeights.reduce(0, +) > 0 else { return nil }
        }
        return CoreMLEnsembleCellCounter(
            specs: selected,
            modelWeights: weights,
            mergeStrategy: strategy
        )
    }

    /// Prefer EfficientNet-B3 when memory allows, otherwise the strongest available lighter tier.
    static func best() -> CellCounter {
        let counters = availableCounters()
        if DeviceCapability.tierCeiling == .best,
           let recommended = counters.first(where: {
               spec(for: $0.identifier)?.isRecommended == true
           }) {
            return recommended
        }
        return counters.first { $0.tier <= DeviceCapability.tierCeiling }
            ?? ClassicalCellCounter()
    }
}
