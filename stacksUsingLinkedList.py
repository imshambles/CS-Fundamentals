class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# [a (self.top), b, c, d]
class Stack:
    def __init__(self):
        self.top = None
        self.height = 0

    def push(self, value):
        newNode = Node(value)
        newNode.next = self.top
        self.top = newNode
        self.height += 1

    def pop(self):
        if self.height <= 0:
            raise Exception("Stack is Empty")
        poppedItem = self.top
        self.top = self.top.next
        self.height -= 1
        return poppedItem.value
    
    def peek(self):
        if self.top is None:
            return None
        return self.top.value

    def isEmpty(self):
        return self.height == 0

    def size(self):
        return self.height

    def __str__(self):
        if self.isEmpty():
            return "Stack is empty"
        
        result = []
        current = self.top
        
        if current:
            result.append(f"{current.value} (top)")
            current = current.next

        while current:
            result.append(str(current.value))
            current = current.next
            
        return " -> ".join(result)

if __name__ == "__main__":
    my_stack = Stack()
    my_stack.push(10)
    my_stack.push(20)
    my_stack.push(30)

    print(f"Current Stack: {my_stack}") # Output: Current Stack: 30 (top) -> 20 -> 10
    print(f"Size: {my_stack.size()}")     # Output: Size: 3
    print(f"Peek: {my_stack.peek()}")     # Output: Peek: 30

    popped_value = my_stack.pop()
    print(f"\nPopped value: {popped_value}") # Output: Popped value: 30
    print(f"Stack after pop: {my_stack}") # Output: Stack after pop: 20 (top) -> 10

    popped_value = my_stack.pop()
    print(f"\nPopped value: {popped_value}") # Output: Popped value: 20
    print(f"Stack after pop: {my_stack}") # Output: Stack after pop: 10 (top)