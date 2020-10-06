#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import namedtuple


# In[2]:


class SparseFeat(namedtuple('SparseFeat',['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int64", embedding_name=None):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print("Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,embedding_name)

    def __hash__(self):
        return self.name.__hash__()


# In[3]:


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float64"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()

