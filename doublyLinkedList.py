class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def append(self, value):
        newNode = Node(value)
        if not self.head:
            self.head = self.tail = newNode
        else:
            self.tail.next = newNode
            newNode.prev = self.tail
            self.tail = newNode
        self.length += 1

    def prepend(self, value):
        newNode = Node(value)
        if not self.head:
            self.head = self.tail = newNode
        else:
            newNode.next = self.head
            self.head.prev = newNode
            self.head = newNode
        self.length += 1

    def insert(self, index, value):
        if index < 0 or index > self.length:
            raise Exception("Index out of bounds")
        # handle head and tail separately
        newNode = Node(value)
        if index == 0:
            if not self.head:
                self.head = self.tail = newNode
            else:
                newNode.next = self.head
                self.head.prev = newNode
                self.head = newNode
        elif index == self.length:
            # Insert at the end
            self.tail.next = newNode
            newNode.prev = self.tail
            self.tail = newNode
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            newNode.next = current.next
            newNode.prev = current
            current.next.prev = newNode
            current.next = newNode
        self.length += 1


    def delete(self, index):
        if index < 0 or index >= self.length:
            raise Exception("Index out of bounds")
        # handle head and tail separately
        if index == 0:
            if self.head.next:
                self.head = self.head.next
                self.head.prev = None
            else:
                self.head = self.tail = None
        else:
            current = self.head
            for _ in range(index):
                current = current.next
            prev_node = current.prev
            next_node = current.next
            prev_node.next = next_node
            if next_node:
                next_node.prev = prev_node
            else:
                self.tail = prev_node
        self.length -= 1

    def get(self, index):
        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds")
        current = self.head
        for _ in range(index):
            current = current.next
        return current.value

    def set(self, index, value):
        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds")
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
    dll = DoublyLinkedList()
    dll.append(10)
    dll.append(20)
    dll.prepend(5)
    print(dll)  # Output: [5, 10, 20]
    dll.insert(1, 7)
    print(dll)  # Output: [5, 7, 10, 20]
    dll.delete(2)
    print(dll)  # Output: [5, 7, 20]
    print(dll.get(1))  # Output: 7
    dll.set(1, 99)
    print(dll)  # Output: [5, 99, 20]