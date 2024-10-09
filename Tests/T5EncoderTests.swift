import MLX
import MLXFast
import MLXNN
import XCTest

@testable import FluxSwift

final class T5EncoderTests: XCTestCase {
  func testT5Encoder() {
    let config = T5Configuration()
    let encoder = T5Encoder(config)
    let tokens = MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5)

    let output = encoder(tokens)
    XCTAssertEqual(output.shape, [1, 5, config.dModel])
  }
}