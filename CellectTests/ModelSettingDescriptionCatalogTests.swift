import XCTest
@testable import Cellect

final class ModelSettingDescriptionCatalogTests: XCTestCase {
    func testEveryAdjustableSettingHasPurposeAndOutcomeGuidance() {
        XCTAssertEqual(Set(ModelSettingKey.allCases.map(\.id)).count, ModelSettingKey.allCases.count)

        for key in ModelSettingKey.allCases {
            let guidance = ModelSettingDescriptionCatalog.guidance(for: key)
            XCTAssertFalse(guidance.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            XCTAssertFalse(guidance.purpose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            XCTAssertFalse(guidance.effects.isEmpty, "\(key) has no outcome explanation")
            XCTAssertGreaterThan(
                guidance.helpText.count,
                guidance.purpose.count,
                "\(key) should explain what changing the setting does"
            )
        }
    }
}
