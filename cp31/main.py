import sys
import math
from collections import defaultdict
# def simple_sieve(limit):
#     """Generate primes up to limit using basic sieve."""
#     is_prime = [True] * (limit + 1)
#     is_prime[0] = is_prime[1] = False
    
#     for p in range(2, int(math.sqrt(limit)) + 1):
#         if is_prime[p]:
#             for i in range(p * p, limit + 1, p):
#                 is_prime[i] = False
    
#     return [i for i in range(limit + 1) if is_prime[i]]


def solve(n, s):
    print(s)


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    t = int(next(it))
    # nint = lambda: int(next(it))
    # niter = lambda: next(it)
    out = []
    for _ in range(t):
        n = int(next(it))
        # s = next(it).decode()
        s = ''
        for i in range(n):
            s += next(it).decode()
        out.append(str(solve(n, s)))
        # out.append(' '.join(map(str, solve(n, p))))
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
