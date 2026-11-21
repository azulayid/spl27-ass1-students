#include "SchedulingPolicy.h"
#include "Request.h"
#include "GPU.h"
#include "Model.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace {

std::vector<int> makeIndices(std::size_t n) {
    std::vector<int> indices(n);
    for (std::size_t i = 0; i < n; ++i) {
        indices[i] = static_cast<int>(i);
    }
    return indices;
}

std::size_t maxCachedPrefixTokens(const Request* req, const std::vector<GPU>& gpus) {
    std::vector<std::string> tokens = req->getCacheableTokens();
    if (tokens.empty()) {
        return 0;
    }

    const std::string modelId = req->getModel().getId();
    const std::size_t embeddingSize = req->getModel().getEmbeddingSize();

    std::size_t best = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        PrefixMatch match = gpus[i].findLongestCachedPrefix(modelId, embeddingSize, tokens);
        if (match.matchedTokens > best) {
            best = match.matchedTokens;
        }
    }
    return best;
}

} // namespace

// =====================
// ArrivalOrderPolicy
// =====================

std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible,
                                            const std::vector<GPU>& gpus) const {
    (void)gpus;
    std::vector<int> order = makeIndices(eligible.size());

    std::sort(order.begin(), order.end(), [&eligible](int a, int b) {
        const Request* ra = eligible[static_cast<std::size_t>(a)];
        const Request* rb = eligible[static_cast<std::size_t>(b)];
        if (ra->getAdmissionCycle() != rb->getAdmissionCycle()) {
            return ra->getAdmissionCycle() < rb->getAdmissionCycle();
        }
        return ra->getId() < rb->getId();
    });

    return order;
}

SchedulingPolicy* ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}

// =====================
// ShortestJobPolicy
// =====================

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible,
                                           const std::vector<GPU>& gpus) const {
    std::vector<int> order = makeIndices(eligible.size());

    std::sort(order.begin(), order.end(), [&eligible, &gpus](int a, int b) {
        const Request* ra = eligible[static_cast<std::size_t>(a)];
        const Request* rb = eligible[static_cast<std::size_t>(b)];

        std::size_t blocksA = gpus.empty()
                                  ? ra->getMaxNeededMemory()
                                  : gpus[0].getBlocksForMemory(ra->getMaxNeededMemory());
        std::size_t blocksB = gpus.empty()
                                  ? rb->getMaxNeededMemory()
                                  : gpus[0].getBlocksForMemory(rb->getMaxNeededMemory());

        if (blocksA != blocksB) {
            return blocksA < blocksB;
        }
        return ra->getId() < rb->getId();
    });

    return order;
}

SchedulingPolicy* ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}

// =====================
// LongestPrefixPolicy
// =====================

std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible,
                                             const std::vector<GPU>& gpus) const {
    std::vector<int> order = makeIndices(eligible.size());

    std::vector<std::size_t> cachedTokens(eligible.size(), 0);
    for (std::size_t i = 0; i < eligible.size(); ++i) {
        cachedTokens[i] = maxCachedPrefixTokens(eligible[i], gpus);
    }

    std::sort(order.begin(), order.end(), [&eligible, &cachedTokens](int a, int b) {
        std::size_t ha = cachedTokens[static_cast<std::size_t>(a)];
        std::size_t hb = cachedTokens[static_cast<std::size_t>(b)];
        if (ha != hb) {
            return ha > hb; // Higher cached prefix comes first
        }
        return eligible[static_cast<std::size_t>(a)]->getId() <
               eligible[static_cast<std::size_t>(b)]->getId();
    });

    return order;
}

SchedulingPolicy* LongestPrefixPolicy::clone() const {
    return new LongestPrefixPolicy(*this);
}
