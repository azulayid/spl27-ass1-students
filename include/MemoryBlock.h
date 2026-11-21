#ifndef MEMORY_BLOCK_H_
#define MEMORY_BLOCK_H_

#include <cstddef>

enum class IdleAction
{
    KeepCached,
    Destroy
};

class MemoryBlock
{
protected:
    size_t size;
    int refCount;
    int *data;

public:
    MemoryBlock(size_t size);
    virtual ~MemoryBlock();
    MemoryBlock(const MemoryBlock &other);
    MemoryBlock &operator=(const MemoryBlock &other);
    MemoryBlock &operator=(MemoryBlock &&other) noexcept;
    MemoryBlock(MemoryBlock &&other) noexcept;

    int readFromBlock(std::size_t offset) const;
    void writeToBlock(std::size_t offset, int value);
    void acquire();
    IdleAction release();

    virtual IdleAction onLastRelease() const = 0;
    virtual MemoryBlock *clone() const = 0;
};

class KVBlock : public MemoryBlock
{
public:
    KVBlock(size_t size) : MemoryBlock(size) {};
    virtual IdleAction onLastRelease() const override { return IdleAction::KeepCached; }
    virtual MemoryBlock *clone() const override { return new KVBlock(*this); }
};

class EmbeddingWorkspace : public MemoryBlock
{
public:
    EmbeddingWorkspace(size_t size) : MemoryBlock(size) {};
    virtual IdleAction onLastRelease() const override { return IdleAction::Destroy; }
    virtual MemoryBlock *clone() const override { return new EmbeddingWorkspace(*this); }
};

#endif