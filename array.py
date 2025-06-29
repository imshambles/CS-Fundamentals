class Array:
    def __init__(self, size):
        self.size = size
        self.length = 0
        self.array = [None]*self.size

    def insert(self, index, value):
        print(index, value)
        if self.length >= self.size:
            raise Exception("Array is full!")
        if index < 0 or index > self.length:
            raise Exception("Index out of bounds!")
        # shifting elements to the right after inserting
        for i in range(self.length, index, -1):
            self.array[i] = self.array[i-1]
        self.array[index] = value
        self.length += 1 

    def append(self, value):
        self.insert(self.length, value)

    def delete(self, index):
        if index < 0 or index >= self.length:
            raise Exception("Index out of Bounds!")
        #shifting elements to the left after deleting
        for i in range(index, self.length-1):
            self.array[i] = self.array[i+1]
        self.array[self.length - 1] = None
        self.length -= 1

    def get(self, index):
        if index < 0 or index >= self.length:
            raise Exception("Index out of Bounds!")
        return self.array[index]

    def set(self, index, value):
        if index < 0 or index >= self.length:
            raise Exception("Index out of Bounds!")
        self.array[index] = value

    def __str__(self):
        return str(self.array[i] for i in range(self.length))

if __name__ == "__main__":
    arr = Array(5)
    arr.append(10)
    arr.append(20)
    arr.insert(1, 15)
    print(arr)  
    arr.delete(1)
    print(arr)  
    print(arr.get(1))  
    arr.set(1, 25)
    print(arr)  

