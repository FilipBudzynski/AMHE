def count_calls(func):
    def wrapper(x):
        wrapper.call_count += 1
        return func(x)

    wrapper.call_count = 0
    return wrapper
