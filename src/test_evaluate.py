from evaluate import normalize_text, compute_cer, compute_wer, compute_accuracy


def main():
    reference = "Hello, World! This is a test."
    prediction = "hello world this is test"

    print("Normalized reference :", normalize_text(reference))
    print("Normalized prediction:", normalize_text(prediction))
    print("CER:", compute_cer(prediction, reference))
    print("WER:", compute_wer(prediction, reference))
    print("Accuracy:", compute_accuracy(prediction, reference))


if __name__ == "__main__":
    main()