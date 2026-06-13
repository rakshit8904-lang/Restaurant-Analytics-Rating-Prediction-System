"""
Custom Hash Table Implementation with Separate Chaining
=======================================================
Used for efficient frequency counting of:
  - City frequencies
  - Cuisine frequencies
  - Country frequencies

Collision handling: Separate Chaining (each bucket is a linked list)
"""

# ─────────────────────────────────────────────
# Linked List Node (for chaining)
# ─────────────────────────────────────────────
class Node:
    def __init__(self, key, value):
        self.key   = key
        self.value = value
        self.next  = None   # pointer to next node in the chain


# ─────────────────────────────────────────────
# Hash Table
# ─────────────────────────────────────────────
class HashTable:
    """
    A hash table that uses separate chaining for collision resolution.

    Parameters
    ----------
    capacity : int
        Number of buckets in the table (default 64).
    """

    def __init__(self, capacity: int = 64):
        self.capacity  = capacity
        self.size      = 0                        # number of unique keys stored
        self.buckets   = [None] * self.capacity   # array of linked-list heads
        self.collisions = 0                       # collision counter (for demo)

    # ──────────────────────────────────────────
    # Hash Function  (polynomial rolling hash)
    # ──────────────────────────────────────────
    def _hash(self, key: str) -> int:
        """
        Polynomial rolling hash.
        h = sum( ord(char) * PRIME^i ) mod capacity

        Prime base = 31 gives a good distribution for lowercase strings.
        """
        PRIME  = 31
        MOD    = self.capacity
        h      = 0
        power  = 1
        for ch in str(key).lower():
            h     = (h + (ord(ch) - ord('a') + 1) * power) % MOD
            power = (power * PRIME) % MOD
        return h

    # ──────────────────────────────────────────
    # Insert / Update
    # ──────────────────────────────────────────
    def insert(self, key: str, value) -> None:
        """
        Insert a key-value pair.
        If the key already exists, update its value.
        Detects collisions and increments counter.
        """
        index  = self._hash(key)
        head   = self.buckets[index]

        # Collision detected: bucket already occupied
        if head is not None:
            # Check if key already present -> update
            current = head
            while current:
                if current.key == key:
                    current.value = value
                    return
                current = current.next
            # Key not found -> true collision (different key, same bucket)
            self.collisions += 1

        # Prepend new node
        new_node      = Node(key, value)
        new_node.next = self.buckets[index]
        self.buckets[index] = new_node
        self.size    += 1

    # ──────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────
    def search(self, key: str):
        """Return the value for *key*, or None if not found."""
        index   = self._hash(key)
        current = self.buckets[index]
        while current:
            if current.key == key:
                return current.value
            current = current.next
        return None

    # ──────────────────────────────────────────
    # Delete
    # ──────────────────────────────────────────
    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if deleted, False if not found."""
        index   = self._hash(key)
        current = self.buckets[index]
        prev    = None

        while current:
            if current.key == key:
                if prev:
                    prev.next = current.next
                else:
                    self.buckets[index] = current.next
                self.size -= 1
                return True
            prev    = current
            current = current.next
        return False

    # ──────────────────────────────────────────
    # Increment (convenience for frequency counting)
    # ──────────────────────────────────────────
    def increment(self, key: str) -> None:
        """Increment the count for *key* by 1. Inserts with 1 if absent."""
        existing = self.search(key)
        if existing is None:
            self.insert(key, 1)
        else:
            self.insert(key, existing + 1)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────
    def get_all(self) -> dict:
        """Return all key-value pairs as a plain Python dict."""
        result = {}
        for head in self.buckets:
            current = head
            while current:
                result[current.key] = current.value
                current = current.next
        return result

    def top_n(self, n: int = 10) -> list:
        """Return top-n (key, value) pairs sorted by value descending."""
        all_items = list(self.get_all().items())
        all_items.sort(key=lambda x: x[1], reverse=True)
        return all_items[:n]

    def load_factor(self) -> float:
        return self.size / self.capacity

    def collision_report(self) -> str:
        occupied      = sum(1 for b in self.buckets if b is not None)
        chained_slots = sum(
            1 for b in self.buckets
            if b is not None and b.next is not None
        )
        return (
            f"Capacity        : {self.capacity}\n"
            f"Unique keys     : {self.size}\n"
            f"Load factor     : {self.load_factor():.3f}\n"
            f"Occupied buckets: {occupied}\n"
            f"Buckets w/ chains (collisions): {chained_slots}\n"
            f"Total collision events        : {self.collisions}\n"
        )

    def __repr__(self):
        return f"HashTable(capacity={self.capacity}, size={self.size})"


# ─────────────────────────────────────────────
# Build frequency tables from a pandas Series
# ─────────────────────────────────────────────
def build_frequency_table(series, capacity: int = 128) -> HashTable:
    """
    Build a HashTable of value frequencies from a pandas Series.
    Handles multi-valued cells separated by commas (e.g. cuisines).
    """
    ht = HashTable(capacity=capacity)
    for value in series.dropna():
        for item in str(value).split(","):
            ht.increment(item.strip())
    return ht


# ─────────────────────────────────────────────
# Demo / self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Hash Table Demo - Separate Chaining")
    print("=" * 50)

    ht = HashTable(capacity=8)   # tiny capacity -> easy to force collisions

    cities = [
        "New Delhi", "Mumbai", "Bangalore", "New Delhi",
        "Chennai", "Hyderabad", "Mumbai", "Kolkata",
        "New Delhi", "Bangalore", "Pune", "Ahmedabad",
    ]

    for city in cities:
        ht.increment(city)

    print("\nAll city frequencies:", ht.get_all())
    print("\nTop 5 cities:", ht.top_n(5))
    print("\nSearch 'Mumbai':", ht.search("Mumbai"))
    print("Search 'Pune'  :", ht.search("Pune"))

    print("\nDelete 'Pune':", ht.delete("Pune"))
    print("After delete, search 'Pune':", ht.search("Pune"))

    print("\nCollision Report:\n" + ht.collision_report())
