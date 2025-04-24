"""
Bulk operations utilities for MongoDB.

This module provides utility functions for performing bulk operations with MongoDB,
which can significantly improve performance when inserting or updating many documents.
"""

import pymongo


def bulk_insert_documents(collection, documents, ordered=False):
    """
    Perform bulk insert of documents into a MongoDB collection.
    
    Args:
        collection: MongoDB collection
        documents: List of documents to insert
        ordered: Whether to perform operations in order and stop on first error
        
    Returns:
        Number of documents inserted
    """
    if not documents:
        return 0
        
    operations = [pymongo.InsertOne(doc) for doc in documents]
    result = collection.bulk_write(operations, ordered=ordered)
    return result.inserted_count


def bulk_upsert_documents(collection, documents, key_fields, ordered=False):
    """
    Perform bulk upsert of documents into a MongoDB collection.
    
    Args:
        collection: MongoDB collection
        documents: List of documents to upsert
        key_fields: List of field names to use as the unique key
        ordered: Whether to perform operations in order and stop on first error
        
    Returns:
        Dictionary with counts of modified, upserted, etc.
    """
    if not documents:
        return {"modified": 0, "upserted": 0}
    
    operations = []
    for doc in documents:
        # Create query from key fields
        query = {field: doc[field] for field in key_fields if field in doc}
        operations.append(pymongo.UpdateOne(
            query,
            {"$setOnInsert": doc},
            upsert=True
        ))
    
    result = collection.bulk_write(operations, ordered=ordered)
    return {
        "modified": result.modified_count,
        "upserted": len(result.upserted_ids) if hasattr(result, "upserted_ids") else 0
    }


def bulk_update_documents(collection, updates, upsert=False, ordered=False):
    """
    Perform bulk updates to a MongoDB collection.
    
    Args:
        collection: MongoDB collection
        updates: List of tuples (query, update_doc)
        upsert: Whether to insert if document doesn't exist
        ordered: Whether to perform operations in order and stop on first error
        
    Returns:
        Dictionary with counts of modified, upserted, etc.
    """
    if not updates:
        return {"modified": 0, "upserted": 0}
        
    operations = [
        pymongo.UpdateOne(query, update_doc, upsert=upsert) 
        for query, update_doc in updates
    ]
    
    result = collection.bulk_write(operations, ordered=ordered)
    return {
        "modified": result.modified_count,
        "upserted": len(result.upserted_ids) if hasattr(result, "upserted_ids") else 0
    }