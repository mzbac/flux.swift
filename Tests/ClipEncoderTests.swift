import MLX
import MLXFast
import MLXNN
import XCTest

@testable import FluxSwift

final class ClipEncoderTests: XCTestCase {
  func testClipEncoder() {
    let config = CLIPConfiguration()
    let encoder = ClipTextModel(config)
    let tokens = MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5)

    let output = encoder(tokens)
    XCTAssertEqual(output.shape, [1, 768])
  }
}