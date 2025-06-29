class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.length = 0

    def append(self, value):
        newNode = Node(value)
        if not self.head:
            self.head = newNode
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = newNode
        self.length += 1

    def prepend(self, value):
        newNode = Node(value)
        newNode.next = self.head
        self.head = newNode
        self.length += 1

    def insert(self, index, value):
        if index < 0 or index > self.length:
            raise Exception("Index out of bounds")
        if index == 0:
            self.prepend(value)
            return
        newNode = Node(value)
        #shift the list to the right
        if not self.head:
            self.head = newNode
        current = self.head
        for _ in range(index-1):
            current = current.next
        newNode.next = current.next
        current.next = newNode
        self.length += 1

    def delete(self, index):
        if index < 0 or index > self.length:
            raise Exception("Index out of bounds")
        if index == 0:
            self.head = self.head.next
            self.length -= 1
        # shift to left
        else:
            current = self.head
            for _ in range(index-1):
                current = current.next
            current.next = current.next.next
        self.length -= 1

    def get(self, index):
        if index < 0 or index > self.length:
            raise Exception("Index out of Bounds")
        current = self.head
        for _ in range(index):
            current = current.next
        return current.value

    def set(self, index, value):
        if index < 0 or index > self.length:
            raise Exception("Index out of bounds")
        current = self.head
        for _ in range(index):
            current = current.next
        current.value = value

    def __str__(self):
        values = []
        current = self.head
        while current:
            values.append(current.value)
            current = current.next
        return str(values)
    
if __name__ == "__main__":
    ll = SinglyLinkedList()
    ll.append(10)
    ll.append(20)
    ll.prepend(5)
    print(ll)  # Output: [5, 10, 20]
    ll.insert(1, 7)
    print(ll)  # Output: [5, 7, 10, 20]
    ll.delete(2)
    print(ll)  # Output: [5, 7, 20]
    print(ll.get(1))  # Output: 7
    ll.set(1, 99)
    print(ll)  # Output: [5, 99, 20]