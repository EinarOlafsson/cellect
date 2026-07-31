import XCTest
@testable import Cellect

final class ModelComparisonPlanTests: XCTestCase {
    private let trained = ComparisonModelChoice(
        id: "trained:model/one",
        displayName: "Trained One",
        isTrainedModel: true
    )
    private let classical = ComparisonModelChoice(
        id: "classical",
        displayName: "Classical CV",
        isTrainedModel: false
    )

    func testQuickPlanIsDeterministicAndKeepsSuppliedModelOrder() {
        var baseline = CountOptions()
        baseline.ensembleModelIdentifiers = ["member-a", "member-b"]
        baseline.ensembleModelWeights = ["member-a": 2, "member-b": 1]

        let first = ModelComparisonPlanner.makePlan(
            models: [classical, trained],
            baseline: baseline,
            preset: .quick,
            includeInvertedInput: true
        )
        let second = ModelComparisonPlanner.makePlan(
            models: [classical, trained],
            baseline: baseline,
            preset: .quick,
            includeInvertedInput: true
        )

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.count, 13)
        XCTAssertEqual(Array(first.prefix(5).map(\.modelIdentifier)),
                       [String](repeating: classical.id, count: 5))
        XCTAssertEqual(Array(first.dropFirst(5).map(\.modelIdentifier)),
                       [String](repeating: trained.id, count: 8))
        XCTAssertEqual(first.map(\.ordinal), Array(0..<first.count))
    }

    func testEveryModelStartsWithNormalizedBaseline() throws {
        var baseline = CountOptions()
        baseline.minAreaPixels = 137
        baseline.maxAreaPixels = 9_001
        baseline.foregroundProbabilityThreshold = 0.57
        baseline.computeMode = .cpuOnly
        baseline.ensembleModelIdentifiers = ["member-a", "member-b"]
        baseline.ensembleModelWeights = ["member-a": 3, "member-b": 1]
        baseline.ensembleMergeStrategy = .maximum

        var expected = baseline
        expected.ensembleModelIdentifiers = []
        expected.ensembleModelWeights = [:]

        let plan = ModelComparisonPlanner.makePlan(
            models: [trained, classical],
            baseline: baseline,
            preset: .quick,
            includeInvertedInput: false
        )

        for model in [trained, classical] {
            let firstForModel = try XCTUnwrap(
                plan.first(where: { $0.modelIdentifier == model.id })
            )
            XCTAssertEqual(firstForModel.variantName, "Baseline")
            XCTAssertEqual(firstForModel.options, expected)
            XCTAssertTrue(firstForModel.changes.isEmpty)
            XCTAssertEqual(firstForModel.summary, "Current settings")
        }
        XCTAssertTrue(plan.allSatisfy { $0.options.ensembleModelIdentifiers.isEmpty })
        XCTAssertTrue(plan.allSatisfy { $0.options.ensembleModelWeights.isEmpty })
    }

    func testQuickAndThoroughCandidateCountsAreBounded() {
        let baseline = CountOptions()

        XCTAssertEqual(
            ModelComparisonPlanner.makePlan(
                models: [trained],
                baseline: baseline,
                preset: .quick,
                includeInvertedInput: false
            ).count,
            7
        )
        XCTAssertEqual(
            ModelComparisonPlanner.makePlan(
                models: [trained],
                baseline: baseline,
                preset: .quick,
                includeInvertedInput: true
            ).count,
            8
        )
        XCTAssertEqual(
            ModelComparisonPlanner.makePlan(
                models: [trained],
                baseline: baseline,
                preset: .thorough,
                includeInvertedInput: false
            ).count,
            28
        )
        XCTAssertEqual(
            ModelComparisonPlanner.makePlan(
                models: [classical],
                baseline: baseline,
                preset: .quick,
                includeInvertedInput: true
            ).count,
            5,
            "Input inversion applies only to trained models."
        )
        XCTAssertEqual(
            ModelComparisonPlanner.makePlan(
                models: [classical],
                baseline: baseline,
                preset: .thorough,
                includeInvertedInput: false
            ).count,
            13
        )
    }

    func testDuplicateOptionSetsAreRemovedAndFirstVariantWins() throws {
        var baseline = CountOptions()
        baseline.foregroundProbabilityThreshold = 0.35

        let plan = ModelComparisonPlanner.makePlan(
            models: [trained],
            baseline: baseline,
            preset: .quick,
            includeInvertedInput: false
        )

        XCTAssertEqual(plan.count, 6)
        XCTAssertEqual(plan.first?.variantName, "Baseline")
        XCTAssertFalse(plan.contains { $0.variantName == "Permissive foreground" })
        for index in plan.indices {
            XCTAssertFalse(
                plan[..<index].contains { $0.options == plan[index].options },
                "Candidate \(index) duplicates an earlier option set."
            )
        }
        XCTAssertEqual(Set(plan.map(\.id)).count, plan.count)
    }

    func testCandidateIDsAndOrdinalsStayUniqueWithDuplicateModelIdentifiers() {
        let duplicate = ComparisonModelChoice(
            id: trained.id,
            displayName: "Same Identifier, Different Choice",
            isTrainedModel: true
        )
        let plan = ModelComparisonPlanner.makePlan(
            models: [trained, duplicate],
            baseline: CountOptions(),
            preset: .quick,
            includeInvertedInput: false
        )

        XCTAssertEqual(plan.count, 14)
        XCTAssertEqual(Set(plan.map(\.id)).count, plan.count)
        XCTAssertEqual(Set(plan.map(\.ordinal)).count, plan.count)
        XCTAssertEqual(plan.map(\.ordinal), Array(0..<plan.count))
    }

    func testNoModelsProducesNoCandidates() {
        XCTAssertTrue(
            ModelComparisonPlanner.makePlan(
                models: [],
                baseline: CountOptions(),
                preset: .thorough,
                includeInvertedInput: true
            ).isEmpty
        )
    }
}
