import Foundation
import MLX
import MLXNN
import Logging

private let logger = Logger(label: "flux.swift.quantization.QuantizationUtils")

public enum QuantizationUtils {
    
    public static let defaultBits = 4
    
    public static let defaultGroupSize = 64
    
    public static let maxChunkSize = 2_000_000_000
    
    public static let componentNameMappings: [String: String] = [
        "vae": "vae",
        "transformer": "transformer",
        "clipEncoder": "text_encoder",
        "t5Encoder": "text_encoder_2",
        "text_encoder": "text_encoder",
        "text_encoder_2": "text_encoder_2"
    ]
    
    public static func isQuantizedWeight(key: String) -> Bool {
        return key.hasSuffix(".scales") || 
               key.hasSuffix(".biases") || 
               (key.hasSuffix(".weight") && key.contains("quantized"))
    }
    
    public static func componentFromKey(_ key: String) -> String? {
        if key.hasPrefix("vae.") {
            return "vae"
        } else if key.hasPrefix("transformer.") {
            return "transformer"
        } else if key.hasPrefix("text_encoder.") {
            return "text_encoder"
        } else if key.hasPrefix("text_encoder_2.") {
            return "text_encoder_2"
        }
        return nil
    }
    
    public static func removeComponentPrefix(_ key: String) -> String {
        if let component = componentFromKey(key) {
            return String(key.dropFirst(component.count + 1))
        }
        return key
    }
    
    public static func sizeInBytes(_ array: MLXArray) -> Int {
        let elementSize: Int
        switch array.dtype {
        case .uint32:
            elementSize = MemoryLayout<UInt32>.size
        case .float16:
            #if os(iOS)
            elementSize = MemoryLayout<Float16>.size
            #else
            // Float16 is 2 bytes (16 bits)
            elementSize = 2
            #endif
        case .float32:
            elementSize = MemoryLayout<Float32>.size
        case .bfloat16:
            // BFloat16 is not a native Swift type, so we hardcode its size
            // BFloat16: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits (2 bytes)
            elementSize = 2
        default:
            // Default to 2 bytes for other half-precision types
            // MLX may use additional types internally that aren't exposed as Swift types
            elementSize = 2
        }
        return array.size * elementSize
    }
    
    public static func totalSize(_ weights: [String: MLXArray]) -> Int {
        return weights.reduce(0) { sum, param in
            sum + sizeInBytes(param.value)
        }
    }
    
    public static func formatSize(_ bytes: Int) -> String {
        let mb = Double(bytes) / 1_000_000
        if mb < 1000 {
            return String(format: "%.1f MB", mb)
        } else {
            let gb = mb / 1000
            return String(format: "%.2f GB", gb)
        }
    }
    
    public static func transformerQuantizationFilter(path: String, module: Module) -> Bool {
        return module is Linear && (module as? Linear)?.weight.shape[1] ?? 0 > 64
    }
    
    public static func defaultQuantizationFilter(path: String, module: Module) -> Bool {
        return module is Linear
    }
    
    public static func directorySize(at url: URL) throws -> Int {
        let resourceKeys: [URLResourceKey] = [.fileSizeKey, .isDirectoryKey]
        var totalSize: Int = 0
        
        if let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: resourceKeys,
            options: [],
            errorHandler: nil
        ) {
            for case let fileURL as URL in enumerator {
                let resourceValues = try fileURL.resourceValues(forKeys: Set(resourceKeys))
                if resourceValues.isDirectory == false {
                    totalSize += resourceValues.fileSize ?? 0
                }
            }
        }
        
        return totalSize
    }
    
    public static func createDirectory(at url: URL) throws {
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true,
            attributes: nil
        )
    }
    
    public static func validateQuantizationParameters(bits: Int, groupSize: Int) throws {
        guard [4, 8].contains(bits) else {
            throw QuantizationError.invalidBits(bits)
        }
        
        guard groupSize > 0 && groupSize.isMultiple(of: 32) else {
            throw QuantizationError.invalidGroupSize(groupSize)
        }
    }
    
    public static func shouldQuantize(_ module: Module) -> Bool {
        let paramCount = module.parameters().flattened().reduce(0) { sum, param in
            sum + param.1.size
        }
        return paramCount > 1_000_000
    }
}

public enum QuantizationError: LocalizedError {
    case invalidBits(Int)
    case invalidGroupSize(Int)
    case componentNotFound(String)
    case weightsNotFound(String)
    case saveFailed(String)
    case loadFailed(String)
    case metadataMissing
    case invalidQuantizationLevel(Int)
    case incompatibleWeightFormat
    case chunkMismatch(expected: Int, found: Int)
    case versionMismatch(expected: String, found: String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidBits(let bits):
            return "Invalid quantization bits: \(bits). Supported values are 4 and 8."
        case .invalidGroupSize(let size):
            return "Invalid group size: \(size). Must be positive and multiple of 32."
        case .componentNotFound(let name):
            return "Component not found: \(name)"
        case .weightsNotFound(let path):
            return "Weights not found at: \(path)"
        case .saveFailed(let reason):
            return "Failed to save quantized weights: \(reason)"
        case .loadFailed(let reason):
            return "Failed to load quantized weights: \(reason)"
        case .metadataMissing:
            return "Quantization metadata file not found"
        case .invalidQuantizationLevel(let bits):
            return "Invalid quantization level: \(bits) bits"
        case .incompatibleWeightFormat:
            return "Weight format is incompatible with quantized loading"
        case .chunkMismatch(let expected, let found):
            return "Chunk count mismatch: expected \(expected), found \(found)"
        case .versionMismatch(let expected, let found):
            return "Version mismatch: expected \(expected), found \(found)"
        }
    }
}

public struct QuantizationStats {
    public let totalParameters: Int
    public let quantizedParameters: Int
    public let originalSize: Int
    public let quantizedSize: Int
    
    public var compressionRatio: Double {
        return Double(originalSize) / Double(quantizedSize)
    }
    
    public var percentQuantized: Double {
        return Double(quantizedParameters) / Double(totalParameters) * 100
    }
    
    public var spaceSaved: Int {
        return originalSize - quantizedSize
    }
    
    public var description: String {
        return """
        Quantization Statistics:
        - Total parameters: \(totalParameters)
        - Quantized parameters: \(quantizedParameters) (\(String(format: "%.1f", percentQuantized))%)
        - Original size: \(QuantizationUtils.formatSize(originalSize))
        - Quantized size: \(QuantizationUtils.formatSize(quantizedSize))
        - Compression ratio: \(String(format: "%.2fx", compressionRatio))
        - Space saved: \(QuantizationUtils.formatSize(spaceSaved))
        """
    }
}

extension QuantizationUtils {
    private static func copyTokenizers(from sourceDirectory: URL, to destinationDirectory: URL) throws {
        let fileManager = FileManager.default
        
        let tokenizerDirs = [
            ("tokenizer", "tokenizer"),      
            ("tokenizer_2", "tokenizer_2") 
        ]
        
        for (sourceName, destName) in tokenizerDirs {
            let sourceTokenizerPath = sourceDirectory.appendingPathComponent(sourceName)
            let destTokenizerPath = destinationDirectory.appendingPathComponent(destName)
            
            if fileManager.fileExists(atPath: sourceTokenizerPath.path) {
                logger.info("Copying \(sourceName) from \(sourceTokenizerPath.path)")
                
                try createDirectory(at: destTokenizerPath)
                
                let tokenizerFiles = try fileManager.contentsOfDirectory(
                    at: sourceTokenizerPath,
                    includingPropertiesForKeys: nil
                )
                
                var copiedCount = 0
                for file in tokenizerFiles {
                    let destFile = destTokenizerPath.appendingPathComponent(file.lastPathComponent)
                    
                    if fileManager.fileExists(atPath: destFile.path) {
                        try fileManager.removeItem(at: destFile)
                    }
                    
                    try fileManager.copyItem(at: file, to: destFile)
                    copiedCount += 1
                    logger.debug("Copied: \(file.lastPathComponent)")
                }
                
                logger.info("Successfully copied \(copiedCount) files for \(sourceName)")
            } else {
                logger.warning("\(sourceName) directory not found at \(sourceTokenizerPath.path)")
            }
        }
        
        let modelIndexSource = sourceDirectory.appendingPathComponent("model_index.json")
        let modelIndexDest = destinationDirectory.appendingPathComponent("model_index.json")
        
        if fileManager.fileExists(atPath: modelIndexSource.path) {
            if fileManager.fileExists(atPath: modelIndexDest.path) {
                try fileManager.removeItem(at: modelIndexDest)
            }
            try fileManager.copyItem(at: modelIndexSource, to: modelIndexDest)
            logger.info("Copied model_index.json")
        }
    }
    
    public static func extractAllWeights(from module: Module) -> [(String, MLXArray)] {
        logger.info("Starting weight extraction...")
        
        var allWeights: [(String, MLXArray)] = []
        
        let leafModules = module.leafModules().flattened()
        logger.info("Found \(leafModules.count) leaf modules")
        
        for (path, leafModule) in leafModules {
            logger.debug("Processing module at path: \(path)")
            
            if let linear = leafModule as? Linear {
                if let quantizedLinear = linear as? QuantizedLinear {
                    logger.debug("  - Found QuantizedLinear layer")
                    let quantizedWeight = quantizedLinear.weight
                    logger.debug("    Weight dtype: \(quantizedWeight.dtype)")
                    
                    allWeights.append(("\(path).weight", quantizedWeight))
                    allWeights.append(("\(path).scales", quantizedLinear.scales))
                    allWeights.append(("\(path).biases", quantizedLinear.biases))
                    if let bias = quantizedLinear.bias {
                        allWeights.append(("\(path).bias", bias))
                    }
                    
                    logger.debug("    Weight shape: \(quantizedLinear.weight.shape)")
                    logger.debug("    Scales shape: \(quantizedLinear.scales.shape)")
                    logger.debug("    Biases shape: \(quantizedLinear.biases.shape)")
                    logger.debug("    Bits: \(quantizedLinear.bits), GroupSize: \(quantizedLinear.groupSize)")
                } else {
                    logger.debug("  - Found regular Linear layer")
                    allWeights.append(("\(path).weight", linear.weight))
                    if let bias = linear.bias {
                        allWeights.append(("\(path).bias", bias))
                    }
                    
                    logger.debug("    Weight shape: \(linear.weight.shape)")
                }
            } else {
                logger.debug("  - Found non-Linear module: \(type(of: leafModule))")
                let params = leafModule.parameters().flattened()
                for (key, value) in params {
                    let fullKey = path.isEmpty ? key : "\(path).\(key)"
                    allWeights.append((fullKey, value))
                    logger.debug("    Parameter \(key): shape=\(value.shape)")
                }
            }
        }
        
        logger.info("Extraction complete. Total weights: \(allWeights.count)")
        
        let totalSize = allWeights.reduce(0) { sum, weight in
            sum + sizeInBytes(weight.1)
        }
        logger.info("Total size: \(formatSize(totalSize))")
        
        return allWeights
    }
    
    public static func hasQuantizedLayers(in module: Module) -> Bool {
        let leafModules = module.leafModules().flattened()
        
        for (_, leafModule) in leafModules {
            if leafModule is QuantizedLinear {
                return true
            }
        }
        
        return false
    }
    
    public static func countQuantizedLayers(in module: Module) -> Int {
        let leafModules = module.leafModules().flattened()
        var count = 0
        
        for (_, leafModule) in leafModules {
            if leafModule is QuantizedLinear {
                count += 1
            }
        }
        
        return count
    }
}

extension QuantizationUtils {
    private static func readModelType(from modelDirectory: URL?) -> String {
        logger.debug("Attempting to read model type from model_index.json")
        
        guard let modelDirectory = modelDirectory else {
            logger.debug("No model directory provided, using default 'flux'")
            return "flux"
        }
        
        let modelIndexPath = modelDirectory.appendingPathComponent("model_index.json")
        
        guard FileManager.default.fileExists(atPath: modelIndexPath.path) else {
            logger.debug("model_index.json not found at \(modelIndexPath.path), using default 'flux'")
            return "flux"
        }
        
        do {
            let data = try Data(contentsOf: modelIndexPath)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let className = json["_class_name"] as? String {
                logger.info("Found model type from model_index.json: \(className)")
                return className
            }
        } catch {
            logger.error("Failed to read model_index.json: \(error)")
        }
        
        return "flux"
    }
    
    public static func saveQuantizedModel(
        _ flux: FLUXComponents,
        to basePath: URL,
        quantizationBits: Int,
        groupSize: Int = 64,
        modelDirectory: URL? = nil
    ) throws {
        logger.info("Starting to save quantized model to \(basePath.path)")
        logger.info("Quantization: \(quantizationBits)-bit, group size: \(groupSize)")
        
        try createDirectory(at: basePath)
        
        let modelType = readModelType(from: modelDirectory)
        
        let metadata = ModelMetadata(
            quantizationBits: quantizationBits,
            groupSize: groupSize,
            modelType: modelType,
            components: ["vae", "transformer", "text_encoder", "text_encoder_2"]
        )
        try saveMetadata(metadata, to: basePath)
        
        for componentInfo in flux.components {
            logger.info("Saving \(componentInfo.name)...")
            try saveModelComponent(
                componentInfo.module,
                name: componentInfo.prefix,
                basePath: basePath,
                quantizationBits: quantizationBits,
                groupSize: groupSize
            )
        }
        
        if let modelDirectory = modelDirectory {
            logger.info("Copying tokenizer files from original model...")
            try copyTokenizers(from: modelDirectory, to: basePath)
        } else {
            logger.warning("No model directory provided - tokenizers will not be copied")
            logger.warning("The quantized model will require internet access or cached tokenizers to work")
        }
        
        logger.info("Quantized model saved successfully!")
        let totalSize = try directorySize(at: basePath)
        logger.info("Total size: \(formatSize(totalSize))")
    }
    
    private static func saveModelComponent(
        _ model: Module,
        name: String,
        basePath: URL,
        quantizationBits: Int,
        groupSize: Int
    ) throws {
        let componentPath = basePath.appendingPathComponent(name)
        try createDirectory(at: componentPath)
        
        logger.info("Getting parameters for \(name)...")
        
        let hasQuantized = hasQuantizedLayers(in: model)
        if hasQuantized {
            let quantizedCount = countQuantizedLayers(in: model)
            logger.info("Model contains \(quantizedCount) quantized layers")
        }
        
        let flattenedParams = extractAllWeights(from: model)
        
        logger.info("Component \(name) has \(flattenedParams.count) parameters")
        
        let totalSize = flattenedParams.reduce(0) { sum, param in
            sum + sizeInBytes(param.1)
        }
        logger.info("Total size for \(name): \(formatSize(totalSize))")
        
        let chunks = splitIntoChunks(flattenedParams, maxSize: maxChunkSize)
        logger.info("Split into \(chunks.count) chunks")
        
        for (index, chunk) in chunks.enumerated() {
            let chunkName = generateChunkName(for: chunk, index: index, totalChunks: chunks.count, componentName: name)
            let chunkPath = componentPath.appendingPathComponent("\(chunkName).safetensors")
            
            let metadata = QuantizationMetadata(
                bits: quantizationBits,
                groupSize: groupSize,
                modelType: name,
                additionalMetadata: [
                    "chunk_index": String(index),
                    "total_chunks": String(chunks.count),
                    "chunk_name": chunkName
                ]
            )
            
            var weightsDict: [String: MLXArray] = [:]
            for (key, value) in chunk {
                if key.hasSuffix(".weight") && value.dtype == .uint32 {
                    logger.debug("Preserving uint32 dtype for quantized weight: \(key)")
                    weightsDict[key] = value
                } else {
                    weightsDict[key] = value
                }
            }
            
            let chunkSize = calculateChunkSize(chunk)
            logger.info("Saving chunk '\(chunkName)' with \(chunk.count) weights")
            logger.info("Chunk size: \(formatSize(chunkSize))")
            
            try MLX.save(
                arrays: weightsDict,
                metadata: metadata.toSafetensorsMetadata(),
                url: chunkPath
            )
            
            logger.info("Chunk '\(chunkName)' saved successfully")
        }
    }
    
    private static func splitIntoChunks(
        _ parameters: [(String, MLXArray)],
        maxSize: Int
    ) -> [[(String, MLXArray)]] {
        logger.debug("Splitting parameters into chunks (max size: \(formatSize(maxSize)))")
        
        var chunks: [[(String, MLXArray)]] = []
        var currentChunk: [(String, MLXArray)] = []
        var currentSize = 0
        
        for (key, array) in parameters {
            let arraySize = sizeInBytes(array)
            
            logger.debug("Parameter \(key): shape=\(array.shape), size=\(formatSize(arraySize))")
            
            if currentSize + arraySize > maxSize && !currentChunk.isEmpty {
                logger.debug("Chunk full at \(formatSize(currentSize)), creating new chunk")
                chunks.append(currentChunk)
                currentChunk = []
                currentSize = 0
            }
            
            currentChunk.append((key, array))
            currentSize += arraySize
        }
        
        if !currentChunk.isEmpty {
            logger.debug("Final chunk size: \(formatSize(currentSize))")
            chunks.append(currentChunk)
        }
        
        if chunks.isEmpty {
            logger.warning("No parameters to save")
            return [parameters]
        }
        
        return chunks
    }
    
    private static func calculateChunkSize(_ chunk: [(String, MLXArray)]) -> Int {
        return chunk.reduce(0) { sum, param in
            sum + sizeInBytes(param.1)
        }
    }
    
    private static func generateChunkName(for chunk: [(String, MLXArray)], index: Int, totalChunks: Int, componentName: String) -> String {     
        if totalChunks == 1 {
            return "flux1_mlx_model"
        } else {
            return String(format: "flux1_mlx_model-%05d-of-%05d", index + 1, totalChunks)
        }
    }
    
    private static func saveMetadata(_ metadata: ModelMetadata, to basePath: URL) throws {
        let metadataPath = basePath.appendingPathComponent("metadata.json")
        logger.info("Saving metadata to \(metadataPath.lastPathComponent)")
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        encoder.dateEncodingStrategy = .iso8601
        
        let data = try encoder.encode(metadata)
        try data.write(to: metadataPath)
        
        logger.info("Metadata saved successfully")
    }
}

extension QuantizationUtils {
    public static func loadQuantizedModel(
        from basePath: URL
    ) throws -> (weights: [String: MLXArray], info: QuantizedWeightInfo) {
        logger.info("Loading quantized model from \(basePath.path)")
        
        let metadataPath = basePath.appendingPathComponent("metadata.json")
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            logger.error("No metadata.json found at \(metadataPath.path)")
            
            if let contents = try? FileManager.default.contentsOfDirectory(atPath: basePath.path) {
                logger.error("Directory contents: \(contents)")
            }
            
            throw QuantizationError.metadataMissing
        }
        
        let metadata = try loadMetadata(from: metadataPath)
        logger.info("Loaded metadata: \(metadata.quantizationBits)-bit quantization")
        logger.info("Model type: \(metadata.modelType)")
        logger.info("Components: \(metadata.components.joined(separator: ", "))")
        
        var allWeights: [String: MLXArray] = [:]
        
        for component in metadata.components {
            let componentPath = basePath.appendingPathComponent(component)
            if FileManager.default.fileExists(atPath: componentPath.path) {
                logger.info("Loading component: \(component)")
                let startTime = Date()
                
                let componentWeights = try loadComponent(
                    from: componentPath,
                    componentName: component
                )
                
                let loadTime = Date().timeIntervalSince(startTime)
                logger.info("Component \(component) loaded in \(String(format: "%.2f", loadTime))s")
                logger.info("Loaded \(componentWeights.count) weights for \(component)")
                
                allWeights.merge(componentWeights) { _, new in new }
            } else {
                logger.warning("Component \(component) not found at \(componentPath.path)")
            }
        }
        
        let info = QuantizedWeightInfo(
            isQuantized: true,
            quantizationBits: metadata.quantizationBits,
            groupSize: metadata.groupSize,
            metadata: [
                "quantization_level": String(metadata.quantizationBits),
                "flux_swift_version": metadata.fluxSwiftVersion,
                "model_type": metadata.modelType
            ]
        )
        
        logger.info("Successfully loaded \(allWeights.count) total quantized weights")
        return (allWeights, info)
    }
    
    private static func loadComponent(
        from path: URL,
        componentName: String
    ) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        
        let files = try FileManager.default.contentsOfDirectory(
            at: path,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { file1, file2 in
            let name1 = file1.deletingPathExtension().lastPathComponent
            let name2 = file2.deletingPathExtension().lastPathComponent
            
            if let num1 = Int(name1), let num2 = Int(name2) {
                return num1 < num2
            }
            
            let fluxPattern = "-(\\d{5})-of-\\d{5}"
            if let match1 = name1.range(of: fluxPattern, options: .regularExpression),
               let match2 = name2.range(of: fluxPattern, options: .regularExpression) {
                let part1 = String(name1[match1])
                let part2 = String(name2[match2])
                let num1 = part1.dropFirst().prefix(5)
                let num2 = part2.dropFirst().prefix(5)
                if let n1 = Int(num1), let n2 = Int(num2) {
                    return n1 < n2
                }
            }
            
            if let match1 = name1.lastIndex(of: "_"),
               let match2 = name2.lastIndex(of: "_") {
                let suffix1 = String(name1[name1.index(after: match1)...])
                let suffix2 = String(name2[name2.index(after: match2)...])
                if let num1 = Int(suffix1), let num2 = Int(suffix2) {
                    return num1 < num2
                }
            }
            
            return name1 < name2
        }
        
        logger.info("Found \(files.count) safetensors files for \(componentName)")
        
        for file in files {
            logger.info("Loading file: \(file.lastPathComponent)")
            
            let chunkWeights = try MLX.loadArrays(url: file)
            
            logger.info("File contains \(chunkWeights.count) weights")
            
            for (key, value) in chunkWeights {
                let fullKey: String
                if key.hasPrefix("\(componentName).") {
                    fullKey = key
                } else {
                    fullKey = "\(componentName).\(key)"
                }
                
                weights[fullKey] = value
            }
        }
        
        return weights
    }
    
    private static func loadMetadata(from path: URL) throws -> ModelMetadata {
        logger.debug("Loading metadata from \(path.lastPathComponent)")
        
        let data = try Data(contentsOf: path)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        
        let metadata = try decoder.decode(ModelMetadata.self, from: data)
        logger.debug("Metadata loaded successfully")
        
        return metadata
    }
    
    public static func detectQuantization(at path: URL) -> QuantizedWeightInfo? {
        logger.debug("Detecting quantization at \(path.path)")
        
        let metadataPath = path.appendingPathComponent("metadata.json")
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            logger.debug("Found metadata.json, checking if quantized")
            
            if let metadata = try? loadMetadata(from: metadataPath) {
                logger.info("Detected quantized model: \(metadata.quantizationBits)-bit")
                return QuantizedWeightInfo(
                    isQuantized: true,
                    quantizationBits: metadata.quantizationBits,
                    groupSize: metadata.groupSize,
                    metadata: [
                        "quantization_level": String(metadata.quantizationBits),
                        "flux_swift_version": metadata.fluxSwiftVersion,
                        "model_type": metadata.modelType
                    ]
                )
            }
        }
        
        if path.pathExtension == "safetensors" {
            logger.debug("Checking single safetensors file")
            logger.debug("Single file quantization detection not supported in MLX Swift")
            return QuantizedWeightInfo(isQuantized: false)
        }
        
        logger.debug("No quantization detected")
        return QuantizedWeightInfo(isQuantized: false)
    }
}