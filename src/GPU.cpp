#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"

#include <stdexcept>
#include <utility>

// =====================
// Constructor & Rule of 5
// =====================

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum),
      blockSize(blockSize),
      memory(blocksNum, nullptr),
      slotRefCounts(blocksNum, 0),
      hasPrefixKey(blocksNum, false),
      slotModelIds(blocksNum),
      slotPrefixes(blocksNum),
      totalReserved(0),
      requestReservations(),
      cacheHits(0),
      evictions(0) {}

GPU::~GPU() {
    clearMemory();
}

GPU::GPU(const GPU& other)
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(other.blocksNum, nullptr),
      slotRefCounts(other.slotRefCounts),
      hasPrefixKey(other.hasPrefixKey),
      slotModelIds(other.slotModelIds),
      slotPrefixes(other.slotPrefixes),
      totalReserved(other.totalReserved),
      requestReservations(other.requestReservations),
      cacheHits(other.cacheHits),
      evictions(other.evictions) {
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            for (std::size_t r = 0; r < other.slotRefCounts[i]; ++r) {
                memory[i]->acquire();
            }
        }
    }
}

GPU& GPU::operator=(const GPU& other) {
    if (this == &other) {
        return *this;
    }

    clearMemory();

    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    memory.assign(blocksNum, nullptr);
    slotRefCounts = other.slotRefCounts;
    hasPrefixKey = other.hasPrefixKey;
    slotModelIds = other.slotModelIds;
    slotPrefixes = other.slotPrefixes;
    totalReserved = other.totalReserved;
    requestReservations = other.requestReservations;
    cacheHits = other.cacheHits;
    evictions = other.evictions;

    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            for (std::size_t r = 0; r < other.slotRefCounts[i]; ++r) {
                memory[i]->acquire();
            }
        }
    }

    return *this;
}

GPU::GPU(GPU&& other) noexcept
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(std::move(other.memory)),
      slotRefCounts(std::move(other.slotRefCounts)),
      hasPrefixKey(std::move(other.hasPrefixKey)),
      slotModelIds(std::move(other.slotModelIds)),
      slotPrefixes(std::move(other.slotPrefixes)),
      totalReserved(other.totalReserved),
      requestReservations(std::move(other.requestReservations)),
      cacheHits(other.cacheHits),
      evictions(other.evictions) {
    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;
    other.memory.clear();
}

GPU& GPU::operator=(GPU&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    clearMemory();

    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    memory = std::move(other.memory);
    slotRefCounts = std::move(other.slotRefCounts);
    hasPrefixKey = std::move(other.hasPrefixKey);
    slotModelIds = std::move(other.slotModelIds);
    slotPrefixes = std::move(other.slotPrefixes);
    totalReserved = other.totalReserved;
    requestReservations = std::move(other.requestReservations);
    cacheHits = other.cacheHits;
    evictions = other.evictions;

    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;
    other.memory.clear();

    return *this;
}

// =====================
// Core GPU Functions
// =====================

std::size_t GPU::availableSlots() const {
    std::size_t activeBlocks = 0;
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] != nullptr && slotRefCounts[i] > 0) {
            ++activeBlocks;
        }
    }
    if (activeBlocks + totalReserved >= blocksNum) {
        return 0;
    }
    return blocksNum - activeBlocks - totalReserved;
}

std::size_t GPU::getBlocksForMemory(std::size_t mem) const {
    if (mem == 0) {
        return 0;
    }
    return (mem + blockSize - 1) / blockSize;
}

void GPU::reserve(std::size_t n, const Request& request) {
    if (n == 0) {
        return;
    }
    int reqId = request.getId();
    requestReservations[reqId] += n;
    totalReserved += n;
}

void GPU::unreserve(std::size_t n, const Request& request) {
    if (n == 0) {
        return;
    }
    int reqId = request.getId();
    auto it = requestReservations.find(reqId);
    if (it == requestReservations.end()) {
        return;
    }
    std::size_t toRelease = (n <= it->second) ? n : it->second;
    it->second -= toRelease;
    totalReserved -= toRelease;
    if (it->second == 0) {
        requestReservations.erase(it);
    }
}

std::size_t GPU::allocateBlock(const Request& request) {
    int reqId = request.getId();
    auto it = requestReservations.find(reqId);
    if (it == requestReservations.end() || it->second == 0) {
        throw std::runtime_error("GPU::allocateBlock: request has no reserved blocks");
    }

    std::size_t slot = findSlotForNewBlock(nullptr);

    // Double dispatch: calls back addMemoryBlock for the concrete Request type
    request.allocateOneBlock(*this, slot);

    clearSlotMetadata(slot);
    acquire(slot);

    it->second -= 1;
    totalReserved -= 1;
    if (it->second == 0) {
        requestReservations.erase(it);
    }

    return slot;
}

void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    (void)request;
    memory[slot] = new KVBlock(blockSize);
    slotRefCounts[slot] = 0;
    clearSlotMetadata(slot);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    (void)request;
    memory[slot] = new EmbeddingWorkspace(blockSize);
    slotRefCounts[slot] = 0;
    clearSlotMetadata(slot);
}

void GPU::acquire(std::size_t slot) {
    memory[slot]->acquire();
    slotRefCounts[slot] += 1;
    if (hasPrefixKey[slot]) {
        cacheHits += 1;
    }
}

void GPU::release(std::size_t slot) {
    IdleAction action = memory[slot]->release();
    if (slotRefCounts[slot] > 0) {
        slotRefCounts[slot] -= 1;
    }

    if (slotRefCounts[slot] == 0) {
        if (action == IdleAction::Destroy || !hasPrefixKey[slot]) {
            delete memory[slot];
            memory[slot] = nullptr;
            clearSlotMetadata(slot);
        }
    }
}

void GPU::copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks) {
    std::vector<bool> protectedSlots(blocksNum, false);

    // Protect any blocks of the chain that already exist on this GPU
    for (std::size_t srcSlot : blocks) {
        if (srcSlot >= other.blocksNum || other.memory[srcSlot] == nullptr || !other.hasPrefixKey[srcSlot]) {
            continue;
        }
        int existing = findSlotByKey(other.slotModelIds[srcSlot], other.slotPrefixes[srcSlot]);
        if (existing != -1) {
            protectedSlots[static_cast<std::size_t>(existing)] = true;
        }
    }

    // Copy missing blocks of the chain
    for (std::size_t srcSlot : blocks) {
        if (srcSlot >= other.blocksNum || other.memory[srcSlot] == nullptr || !other.hasPrefixKey[srcSlot]) {
            continue;
        }

        const std::string& modelId = other.slotModelIds[srcSlot];
        const std::vector<std::string>& key = other.slotPrefixes[srcSlot];

        if (findSlotByKey(modelId, key) != -1) {
            continue;
        }

        std::size_t destSlot = findSlotForNewBlock(&protectedSlots);
        memory[destSlot] = other.memory[srcSlot]->clone();
        slotRefCounts[destSlot] = 0;
        hasPrefixKey[destSlot] = true;
        slotModelIds[destSlot] = modelId;
        slotPrefixes[destSlot] = key;
        protectedSlots[destSlot] = true;
    }
}

void GPU::assignPrefixToBlock(std::size_t block,
                              const std::vector<std::string>& prefixTokens,
                              const std::string& modelId) {
    if (block >= blocksNum || memory[block] == nullptr) {
        return;
    }

    int existingSlot = findSlotByKey(modelId, prefixTokens);
    if (existingSlot != -1 && static_cast<std::size_t>(existingSlot) != block) {
        clearSlotMetadata(block);
        return;
    }

    hasPrefixKey[block] = true;
    slotModelIds[block] = modelId;
    slotPrefixes[block] = prefixTokens;
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    memory[blockId]->writeToBlock(offset, value);
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId,
                                         std::size_t embeddingSize,
                                         const std::vector<std::string>& promptTokens) const {
    PrefixMatch result;
    result.matchedTokens = 0;

    if (embeddingSize == 0 || blockSize < embeddingSize) {
        return result;
    }

    std::size_t tokensPerBlock = blockSize / embeddingSize;
    for (std::size_t k = tokensPerBlock; k <= promptTokens.size(); k += tokensPerBlock) {
        std::vector<std::string> key(promptTokens.begin(), promptTokens.begin() + k);
        int slotIdx = findSlotByKey(modelId, key);
        if (slotIdx == -1) {
            break;
        }
        result.blockIds.push_back(static_cast<std::size_t>(slotIdx));
        result.matchedTokens = k;
    }

    return result;
}

// =====================
// Helper Getters
// =====================

std::size_t GPU::getBlocksNum() const {
    return blocksNum;
}

std::size_t GPU::getBlockSize() const {
    return blockSize;
}

std::size_t GPU::getCacheHits() const {
    return cacheHits;
}

std::size_t GPU::getEvictions() const {
    return evictions;
}

bool GPU::isSlotIdle(std::size_t slot) const {
    return slot < blocksNum && memory[slot] != nullptr && slotRefCounts[slot] == 0;
}

// =====================
// Private Helpers
// =====================

void GPU::clearMemory() {
    for (std::size_t i = 0; i < memory.size(); ++i) {
        delete memory[i];
        memory[i] = nullptr;
    }
    memory.clear();
}

void GPU::clearSlotMetadata(std::size_t slot) {
    hasPrefixKey[slot] = false;
    slotModelIds[slot].clear();
    slotPrefixes[slot].clear();
}

std::size_t GPU::findSlotForNewBlock(const std::vector<bool>* protectedSlots) {
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] == nullptr) {
            return i;
        }
    }

    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] != nullptr && slotRefCounts[i] == 0) {
            if (protectedSlots != nullptr && (*protectedSlots)[i]) {
                continue;
            }
            delete memory[i];
            memory[i] = nullptr;
            clearSlotMetadata(i);
            evictions += 1;
            return i;
        }
    }

    throw std::runtime_error("GPU: no empty or idle slot available for allocation");
}

int GPU::findSlotByKey(const std::string& modelId,
                       const std::vector<std::string>& prefixTokens) const {
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] != nullptr &&
            hasPrefixKey[i] &&
            slotModelIds[i] == modelId &&
            slotPrefixes[i] == prefixTokens) {
            return static_cast<int>(i);
        }
    }
    return -1;
}
