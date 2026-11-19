#ifndef MODEL_H_
#define MODEL_H_

#include <cstddef>
#include <cstdint>
#include <cctype>
#include <string>
#include <vector>

class Request;
class GPU;

class Model
{
public:
    ~Model() = default;
    Model(const std::string &id, std::size_t embeddingSize);
    Model(const Model &other) = default;
    Model &operator=(const Model &other) = default;
    Model &operator=(Model &&other) noexcept = default;
    Model(Model &&other) noexcept = default;

    const std::string getId() const;
    void setId(const std::string &id);
    size_t getEmbeddingSize() const;
    void setEmbeddingSize(size_t embeddingSize);

    std::vector<std::string> tokenizer(std::string text);
    void generateTokenEmbedding(const std::vector<std::string> &, GPU &, size_t, size_t) const;

private:
    std::string modelId;
    std::size_t embeddingSize;

    std::uint64_t mixByte(std::uint64_t state, unsigned char byte) const;
    std::uint64_t mixString(std::uint64_t state, const std::string &text) const;
};

#endif