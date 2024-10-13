import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest

@testable import FluxSwift

final class MultiModalDiffusionTransformerTests: XCTestCase {
    func testMultiModalDiffusionTransformer() {
        let config = MultiModalDiffusionConfiguration()
        let transformer = MultiModalDiffusionTransformer(config)
        
        let t = 0
        let promptEmbeds = MLXRandom.normal([1, 15, config.jointAttentionDim])
        let pooledPromptEmbeds = MLXRandom.normal([1, config.pooledProjectionDim])
        let hiddenStates = MLXRandom.normal([1, config.jointAttentionDim, config.inChannels])
        
        let evaluateParameters = EvaluateParameters(width: 1024, height: 1024, shiftSigmas: true)
        
        let output = transformer(
            t: t,
            promptEmbeds: promptEmbeds,
            pooledPromptEmbeds: pooledPromptEmbeds,
            hiddenStates: hiddenStates,
            evaluateParameters: evaluateParameters
        )
        
        XCTAssertEqual(output.shape, [1, config.jointAttentionDim, config.inChannels])
    }
}
