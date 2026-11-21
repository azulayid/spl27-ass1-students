#include "Request.h"
#include "Model.h"
#include "GPU.h"

#include <sstream>

// =====================
// Request (Base)
// =====================

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt)
    : id(id),
      model(model),
      admissionCycle(admissionCycle),
      tokens(model.tokenizer(prompt)),
      memoryBlocks(),
      reservedBlocks(0) {}

void Request::deallocateMemory(GPU& gpu) {
    std::size_t toUnreserve = getNeedToUnreserve();
    if (toUnreserve > 0) {
        gpu.unreserve(toUnreserve, *this);
    }
    for (std::size_t i = 0; i < memoryBlocks.size(); ++i) {
        gpu.release(memoryBlocks[i]);
    }
    memoryBlocks.clear();
}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}

// =====================
// GenerativeRequest
// =====================

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle,
                                     std::string prompt, std::string scriptedOutput,
                                     std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt),
      scriptedOutput(scriptedOutput),
      maxOutputTokens(maxOutputTokens),
      outputTokens(),
      nextTokenIndex(0),
      usedReservedBlocks(0) {
    std::vector<std::string> parsed = model.tokenizer(scriptedOutput);
    std::size_t count = (parsed.size() < maxOutputTokens) ? parsed.size() : maxOutputTokens;
    outputTokens.assign(parsed.begin(), parsed.begin() + count);
}

std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens);
    std::size_t remainingTokens = (tokens.size() - match.matchedTokens) + maxOutputTokens;
    return model.getEmbeddingSize() * remainingTokens;
}

std::size_t GenerativeRequest::getMaxNeededMemory() const {
    return model.getEmbeddingSize() * (tokens.size() + maxOutputTokens);
}

std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}

void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

Request* GenerativeRequest::clone(const Model& newModel) const {
    GenerativeRequest* copy = new GenerativeRequest(id, newModel, admissionCycle, "", scriptedOutput, maxOutputTokens);
    copy->tokens = tokens;
    copy->memoryBlocks = memoryBlocks;
    copy->reservedBlocks = reservedBlocks;
    copy->outputTokens = outputTokens;
    copy->nextTokenIndex = nextTokenIndex;
    copy->usedReservedBlocks = usedReservedBlocks;
    return copy;
}

void GenerativeRequest::reserveMemory(GPU& gpu) {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens);
    for (std::size_t i = 0; i < match.blockIds.size(); ++i) {
        gpu.acquire(match.blockIds[i]);
        memoryBlocks.push_back(match.blockIds[i]);
    }
    nextTokenIndex = match.matchedTokens;

    std::size_t remainingTokens = (tokens.size() - match.matchedTokens) + maxOutputTokens;
    std::size_t neededMem = model.getEmbeddingSize() * remainingTokens;
    reservedBlocks = gpu.getBlocksForMemory(neededMem);
    usedReservedBlocks = 0;
    gpu.reserve(reservedBlocks, *this);
}

void GenerativeRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    std::size_t embSize = model.getEmbeddingSize();
    std::size_t tokensPerBlock = gpu.getBlockSize() / embSize;
    std::size_t tokenInBlock = nextTokenIndex % tokensPerBlock;

    if (tokenInBlock == 0) {
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
        usedReservedBlocks += 1;
    }

    std::size_t currentSlot = memoryBlocks.back();
    std::size_t offset = tokenInBlock * embSize;

    // Build prefix from token 0 up to and including nextTokenIndex
    std::vector<std::string> prefix;
    prefix.reserve(nextTokenIndex + 1);
    for (std::size_t i = 0; i <= nextTokenIndex; ++i) {
        if (i < tokens.size()) {
            prefix.push_back(tokens[i]);
        } else {
            prefix.push_back(outputTokens[i - tokens.size()]);
        }
    }

    model.generateTokenEmbedding(prefix, gpu, currentSlot, offset);
    nextTokenIndex += 1;

    // Only full blocks get a prefix key
    if (nextTokenIndex % tokensPerBlock == 0) {
        gpu.assignPrefixToBlock(currentSlot, prefix, model.getId());
    }
}

bool GenerativeRequest::finished() const {
    return nextTokenIndex >= tokens.size() + outputTokens.size();
}

std::size_t GenerativeRequest::getNeedToUnreserve() const {
    if (reservedBlocks <= usedReservedBlocks) {
        return 0;
    }
    return reservedBlocks - usedReservedBlocks;
}

std::string GenerativeRequest::getOutput() const {
    std::string out;
    for (std::size_t i = 0; i < outputTokens.size(); ++i) {
        if (i > 0) {
            out += " ";
        }
        out += outputTokens[i];
    }
    return out;
}

// =====================
// EmbeddingRequest
// =====================

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt),
      nextTokenIndex(0),
      usedReservedBlocks(0),
      meanVector(model.getEmbeddingSize(), 0) {}

std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const {
    (void)gpu;
    return 2 * model.getEmbeddingSize();
}

std::size_t EmbeddingRequest::getMaxNeededMemory() const {
    return 2 * model.getEmbeddingSize();
}

std::vector<std::string> EmbeddingRequest::getCacheableTokens() const {
    return std::vector<std::string>();
}

void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

Request* EmbeddingRequest::clone(const Model& newModel) const {
    EmbeddingRequest* copy = new EmbeddingRequest(id, newModel, admissionCycle, "");
    copy->tokens = tokens;
    copy->memoryBlocks = memoryBlocks;
    copy->reservedBlocks = reservedBlocks;
    copy->nextTokenIndex = nextTokenIndex;
    copy->usedReservedBlocks = usedReservedBlocks;
    copy->meanVector = meanVector;
    return copy;
}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    reservedBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    usedReservedBlocks = 0;
    gpu.reserve(reservedBlocks, *this);

    for (std::size_t i = 0; i < reservedBlocks; ++i) {
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
        usedReservedBlocks += 1;
    }
}

void EmbeddingRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    std::size_t embSize = model.getEmbeddingSize();
    std::size_t blockSize = gpu.getBlockSize();

    std::size_t sumSlot = memoryBlocks[0];
    std::size_t curSlot = memoryBlocks[embSize / blockSize];
    std::size_t curOffset = embSize % blockSize;

    std::vector<std::string> singleToken(1, tokens[nextTokenIndex]);
    model.generateTokenEmbedding(singleToken, gpu, curSlot, curOffset);

    for (std::size_t i = 0; i < embSize; ++i) {
        int updatedSum = gpu.readFromBlock(sumSlot, i) + gpu.readFromBlock(curSlot, curOffset + i);
        gpu.writeToBlock(sumSlot, i, updatedSum);
    }

    nextTokenIndex += 1;

    if (nextTokenIndex == tokens.size()) {
        int count = static_cast<int>(tokens.size());
        for (std::size_t i = 0; i < embSize; ++i) {
            meanVector[i] = gpu.readFromBlock(sumSlot, i) / count;
        }
    }
}

bool EmbeddingRequest::finished() const {
    return nextTokenIndex >= tokens.size();
}

std::size_t EmbeddingRequest::getNeedToUnreserve() const {
    if (reservedBlocks <= usedReservedBlocks) {
        return 0;
    }
    return reservedBlocks - usedReservedBlocks;
}

std::string EmbeddingRequest::getOutput() const {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < meanVector.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << meanVector[i];
    }
    oss << "]";
    return oss.str();
}
