# [d, c, b, a(top)]
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()

    def peek(self):
        if self.isEmpty():
            raise Exception("No items in the stack!")
        return self.items[-1]

    def isEmpty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def __str__(self):
        return str(self.items)

if __name__ == "__main__":
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print(stack)        # Output: [1, 2, 3]
    print(stack.pop())  # Output: 3
    print(stack.peek()) # Output: 2
    print(stack) #Output: [1, 2]