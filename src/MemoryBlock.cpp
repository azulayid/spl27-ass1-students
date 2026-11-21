#include "MemoryBlock.h";

MemoryBlock::MemoryBlock(size_t size) : size(size), refCount(0), data(new int[size]()) {}

MemoryBlock::~MemoryBlock()
{
    delete[] data;
    data = nullptr;
}

MemoryBlock::MemoryBlock(const MemoryBlock &other) : size(other.size), refCount(0), data(nullptr)
{
    if (other.data != nullptr && other.size != 0)
    {
        this->data = new int[other.size];
        for (size_t i = 0; i < other.size; i++)
        {
            this->data[i] = other.data[i];
        }
    }
}

MemoryBlock &MemoryBlock::operator=(const MemoryBlock &other)
{
    if (this != &other)
    {

        if (other.data != nullptr && other.size != 0)
        {
            int *newData = new int[other.size];

            for (size_t i = 0; i < other.size; i++)
            {
                newData[i] = other.data[i];
            }

            delete[] data;
            this->size = other.size;
            this->refCount = 0;
            data = newData;
        }
        else
        {
            this->size = 0;
            this->refCount = 0;
            delete[] data;
            this->data = nullptr;
        }
    }

    return *this;
}

void MemoryBlock::writeToBlock(std::size_t offset, int value)
{
    if (data != nullptr && offset < size)
    {
        data[offset] = value;
    }
}

int MemoryBlock::readFromBlock(std::size_t offset) const
{
    if (data != nullptr && offset < size)
    {
        return data[offset];
    }
}

MemoryBlock &MemoryBlock::operator=(MemoryBlock &&other) noexcept
{
    if (this != &other)
    {
        delete[] data;
        data = other.data;

        this->size = other.size;
        this->refCount = other.refCount;

        other.data = nullptr;
        other.size = 0;
        other.refCount = 0;
    }

    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock &&other)noexcept : size(other.size), refCount(other.refCount), data(other.data)
{
    other.data = nullptr;
    other.refCount = 0;
    other.size = 0;
}

void MemoryBlock::acquire()
{
    refCount++;
}
IdleAction MemoryBlock::release()
{
    if(refCount>0){
        refCount--; 
    }
    
    if(refCount == 0){
        return onLastRelease();
    }

    return IdleAction::KeepCached;
}
