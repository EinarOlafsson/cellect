import Foundation

/// Stable identifiers for every user-adjustable counting control.
///
/// Keeping the descriptions independent of SwiftUI lets the same guidance appear in model
/// settings, comparison-plan setup, and saved comparison-result manifests without duplicating
/// scientific claims in multiple views.
enum ModelSettingKey: String, CaseIterable, Identifiable, Sendable {
    case modelSelection
    case ensembleEnabled
    case ensembleMergeStrategy
    case ensembleModelSelection
    case ensembleWeight
    case minimumArea
    case maximumArea
    case foregroundThreshold
    case touchingSeparation
    case adaptiveBoundary
    case boundaryCutoff
    case meanCellConfidence
    case coreProbabilityLevel
    case minimumCoreFraction
    case minimumPerimeterSupport
    case inputInversion
    case computeHardware
    case classicalPolarity
    case medianDenoise
    case otsuOffset
    case openingIterations
    case closingIterations

    var id: String { rawValue }
}

struct ModelSettingGuidance: Sendable {
    let title: String
    let purpose: String
    let effects: [ModelSettingEffect]

    var helpText: String {
        ([purpose] + effects.map(\.helpText)).joined(separator: " ")
    }
}

enum ModelSettingEffect: Sendable {
    case lowerHigher(lower: String, higher: String)
    case offOn(off: String, on: String)
    case choices(String)

    fileprivate var helpText: String {
        switch self {
        case .lowerHigher(let lower, let higher):
            return "Lower: \(lower) Higher: \(higher)"
        case .offOn(let off, let on):
            return "Off: \(off) On: \(on)"
        case .choices(let explanation):
            return explanation
        }
    }
}

enum ModelSettingDescriptionCatalog {
    static func guidance(for key: ModelSettingKey) -> ModelSettingGuidance {
        switch key {
        case .modelSelection:
            return ModelSettingGuidance(
                title: "Model",
                purpose: "Chooses the algorithm that generates the cell mask.",
                effects: [.choices(
                    "Larger tiers can capture harder boundaries but usually take longer and use more memory; the measured held-out scores are shown for each trained model."
                )]
            )
        case .ensembleEnabled:
            return ModelSettingGuidance(
                title: "Combine multiple trained models",
                purpose: "Controls whether one model or several models generate the probability maps.",
                effects: [.offOn(
                    off: "runs only the selected model.",
                    on: "runs every checked model and can improve agreement, but increases processing time."
                )]
            )
        case .ensembleMergeStrategy:
            return ModelSettingGuidance(
                title: "Merge probabilities",
                purpose: "Controls how the selected models' foreground and boundary probabilities are fused before cells are reconstructed.",
                effects: [.choices(
                    "Weighted average balances evidence; median resists one outlier; maximum is permissive; minimum requires agreement and is conservative."
                )]
            )
        case .ensembleModelSelection:
            return ModelSettingGuidance(
                title: "Ensemble members",
                purpose: "Chooses which trained models contribute to the combined probability map.",
                effects: [.choices(
                    "More members take longer and are helpful only when their errors differ; at least two models are required."
                )]
            )
        case .ensembleWeight:
            return ModelSettingGuidance(
                title: "Model weight",
                purpose: "Sets a model's relative influence in a weighted average; weights are normalized across selected models.",
                effects: [.lowerHigher(
                    lower: "reduces that model's influence.",
                    higher: "makes the result follow that model more closely."
                )]
            )
        case .minimumArea:
            return ModelSettingGuidance(
                title: "Minimum area",
                purpose: "Removes reconstructed objects smaller than this source-image pixel area.",
                effects: [.lowerHigher(
                    lower: "keeps small cells but also more specks and fragments.",
                    higher: "removes more debris but can discard genuinely small cells."
                )]
            )
        case .maximumArea:
            return ModelSettingGuidance(
                title: "Maximum area",
                purpose: "Removes reconstructed objects larger than this source-image pixel area.",
                effects: [
                    .offOn(
                        off: "keeps every object above the minimum area.",
                        on: "rejects objects above the chosen limit."
                    ),
                    .lowerHigher(
                        lower: "removes more large objects or merged clumps.",
                        higher: "keeps more large cells and clumps."
                    ),
                ]
            )
        case .foregroundThreshold:
            return ModelSettingGuidance(
                title: "Foreground confidence",
                purpose: "Sets the minimum model probability for a pixel to be considered part of a cell.",
                effects: [.lowerHigher(
                    lower: "produces larger, more permissive masks and more false positives.",
                    higher: "produces smaller, stricter masks and can miss faint cells."
                )]
            )
        case .touchingSeparation:
            return ModelSettingGuidance(
                title: "Separate touching cells",
                purpose: "Controls whether the contact-boundary output divides connected foreground into instances.",
                effects: [.offOn(
                    off: "counts each connected foreground region as one object.",
                    on: "uses boundary evidence to split touching cells, with some risk of over-splitting."
                )]
            )
        case .adaptiveBoundary:
            return ModelSettingGuidance(
                title: "Adaptive boundary cutoff",
                purpose: "Controls whether one fixed cutoff or a per-image stability sweep selects the touching-cell partition.",
                effects: [.offOn(
                    off: "uses the displayed boundary cutoff exactly.",
                    on: "searches 10–90% and falls back to the preferred cutoff when the partition is not reliably stable."
                )]
            )
        case .boundaryCutoff:
            return ModelSettingGuidance(
                title: "Boundary cutoff",
                purpose: "Sets how much predicted contact-boundary evidence is removed before touching cells are expanded into separate labels.",
                effects: [.lowerHigher(
                    lower: "treats more pixels as boundaries, usually making more or finer splits.",
                    higher: "accepts weaker boundary pixels as cell interior, usually making fewer splits or merged cells."
                )]
            )
        case .meanCellConfidence:
            return ModelSettingGuidance(
                title: "Minimum mean cell confidence",
                purpose: "Requires each reconstructed cell's average foreground probability to reach this value.",
                effects: [
                    .offOn(
                        off: "does not filter cells by mean foreground confidence.",
                        on: "removes cells below the selected average confidence."
                    ),
                    .lowerHigher(
                        lower: "keeps more faint or uncertain objects.",
                        higher: "removes more weak objects but can reject faint real cells."
                    ),
                ]
            )
        case .coreProbabilityLevel:
            return ModelSettingGuidance(
                title: "High-confidence core level",
                purpose: "Defines which pixels count as a cell's high-confidence core; it matters only when a minimum core fraction is enabled.",
                effects: [.lowerHigher(
                    lower: "lets more pixels qualify as core.",
                    higher: "reserves the core for only the strongest foreground predictions."
                )]
            )
        case .minimumCoreFraction:
            return ModelSettingGuidance(
                title: "Minimum high-confidence core",
                purpose: "Requires at least this fraction of each cell to exceed the high-confidence core level.",
                effects: [
                    .offOn(
                        off: "does not filter cells by core fraction.",
                        on: "removes cells whose strong core is too small."
                    ),
                    .lowerHigher(
                        lower: "permits cells with a small confident center.",
                        higher: "requires more of every cell to be strongly predicted."
                    ),
                ]
            )
        case .minimumPerimeterSupport:
            return ModelSettingGuidance(
                title: "Minimum perimeter support",
                purpose: "Requires the mean boundary-head probability along a reconstructed cell's perimeter to reach this value.",
                effects: [
                    .offOn(
                        off: "does not filter cells by perimeter response.",
                        on: "removes cells with weak perimeter response."
                    ),
                    .lowerHigher(
                        lower: "keeps more weakly outlined cells.",
                        higher: "is stricter and may remove isolated cells because this model head primarily learned cell-contact boundaries."
                    ),
                ]
            )
        case .inputInversion:
            return ModelSettingGuidance(
                title: "Invert grayscale input",
                purpose: "Controls whether intensity is sent to the model normally or as one minus intensity.",
                effects: [.offOn(
                    off: "uses the image's original light/dark polarity.",
                    on: "swaps bright and dark; use it when the microscope contrast is opposite to the model's useful polarity."
                )]
            )
        case .computeHardware:
            return ModelSettingGuidance(
                title: "Compute hardware",
                purpose: "Chooses which Apple processors Core ML may use; it is a runtime setting, not a mask-quality setting.",
                effects: [.choices(
                    "Automatic is normally fastest; restricted modes help compare compatibility or troubleshoot a device."
                )]
            )
        case .classicalPolarity:
            return ModelSettingGuidance(
                title: "Cell appearance",
                purpose: "Tells classical thresholding whether cells lie below or above the Otsu intensity cutoff.",
                effects: [.choices(
                    "Choose dark cells for objects darker than their background and bright cells for the reverse."
                )]
            )
        case .medianDenoise:
            return ModelSettingGuidance(
                title: "3 × 3 median denoise",
                purpose: "Controls whether a small median filter is applied before classical thresholding.",
                effects: [.offOn(
                    off: "preserves raw detail and noise.",
                    on: "suppresses isolated camera noise but can erase the finest structures."
                )]
            )
        case .otsuOffset:
            return ModelSettingGuidance(
                title: "Otsu threshold offset",
                purpose: "Adds an intensity offset to the automatically calculated 8-bit Otsu cutoff.",
                effects: [.lowerHigher(
                    lower: "keeps more bright-mode foreground but less dark-mode foreground.",
                    higher: "keeps more dark-mode foreground but less bright-mode foreground."
                )]
            )
        case .openingIterations:
            return ModelSettingGuidance(
                title: "Opening iterations",
                purpose: "Repeatedly erodes then dilates the binary foreground with a 3 × 3 neighborhood.",
                effects: [
                    .offOn(
                        off: "does not apply opening.",
                        on: "removes small islands and thin connections."
                    ),
                    .lowerHigher(
                        lower: "is gentler.",
                        higher: "removes more noise and thin bridges but can erase small cells."
                    ),
                ]
            )
        case .closingIterations:
            return ModelSettingGuidance(
                title: "Closing iterations",
                purpose: "Repeatedly dilates then erodes the binary foreground with a 3 × 3 neighborhood.",
                effects: [
                    .offOn(
                        off: "does not apply closing.",
                        on: "fills small gaps and holes."
                    ),
                    .lowerHigher(
                        lower: "is gentler.",
                        higher: "fills larger gaps but can connect nearby cells."
                    ),
                ]
            )
        }
    }

    static func helpText(for key: ModelSettingKey) -> String {
        guidance(for: key).helpText
    }
}
