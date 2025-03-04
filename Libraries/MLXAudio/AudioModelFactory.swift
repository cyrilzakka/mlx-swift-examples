//
//  File.swift
//  mlx-libraries
//
//  Created by Cyril Zakka on 3/4/25.
//

import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public enum AudioError: Error {
    case textRequired
    case processing(String)
}
