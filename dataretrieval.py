import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firestore app
cred = credentials.Certificate(
    "serviceAccountKey.json")
print(cred)
firebase_admin.initialize_app(cred)
db = firestore.client()


def retrieve_data(collection_name):
    """
    Retrieve data from a Firestore collection.

    Args:
        collection_name (str): The name of the collection to retrieve data from.

    Returns:
        list: A list of dictionaries containing the retrieved documents.
    """
    try:
        # Get all documents from the specified collection
        docs = db.collection(collection_name).get()
        data = []
        # Iterate through the documents and append them to the data list
        for doc in docs:
            data.append(doc.to_dict())
        return data
    except Exception as e:
        print("An error occurred:", e)
        return []


if __name__ == '__main__':
   
    # Example usage:
    collection_name = "CommentTest"
    retrieved_data = retrieve_data(collection_name)
    print("Retrieved data:")
    for document in retrieved_data:
        print(document)
