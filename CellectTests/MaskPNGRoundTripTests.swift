import XCTest
@testable import Cellect

final class MaskPNGRoundTripTests: XCTestCase {
    func testLabel16RoundTripPreservesOrientationAndInstanceIDs() throws {
        let width = 4
        let height = 3
        let labels: [UInt16] = [
            0, 1, 2, 3,
            4, 255, 256, 1_024,
            12, 0, UInt16.max - 1, UInt16.max,
        ]

        let png = try XCTUnwrap(MaskPNG.label16(labels, width: width, height: height))
        let decoded = try XCTUnwrap(MaskPNG.decodeLabel16(png))

        XCTAssertEqual(decoded.width, width)
        XCTAssertEqual(decoded.height, height)
        XCTAssertEqual(decoded.labels, labels)
    }
}
