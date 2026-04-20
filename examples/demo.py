import fast_thompson as ft

def main():
    # Create some items (id, successes, failures)
    items = [
        ft.Item(1, 10, 5),
        ft.Item(2, 3, 8),
        ft.Item(3, 20, 2),
        ft.Item(4, 7, 7),
        ft.Item(5, 15, 10),
    ]

    # Initialize Thompson sampler
    model = ft.Thompson(items)

    # No forbidden items
    forbidden = []

    # Sample top-k items
    result = model.sample(3, forbidden)

    print("Top sampled items:")
    for r in result:
        print(f"id={r.id}, score={r.score:.5f}")

if __name__ == "__main__":
    main()