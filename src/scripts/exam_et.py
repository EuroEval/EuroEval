"""Script for processing the Exam-et dataset into EuroEval format."""

from datasets import load_dataset
from euroeval.tasks import KNOW
from euroeval.languages import ET


def process_exam_et():
    """Process the Exam-et dataset for EuroEval."""
    
    # Load the dataset from Hugging Face
    raw_dataset = load_dataset("TalTechNLP/exam_et")
    
    def format_sample(sample):
        """Format a single sample to EuroEval format."""
        return {
            "id": sample.get("id", ""),
            "question": sample["question"],
            "options": sample["options"],  # Assuming multiple choice format
            "answer": sample["answer"],
            "language": "et",
            "subject": sample.get("subject", "general"),
        }
    
    # Process the dataset
    processed_dataset = {}
    
    # Split the data (adjust based on actual dataset structure)
    if "train" in raw_dataset:
        train_data = raw_dataset["train"].map(format_sample)
        # Create splits with recommended sizes: 1024 train, 256 val, 2048 test
        total_train = len(train_data)
        train_split = train_data.select(range(min(1024, total_train // 2)))
        val_split = train_data.select(range(min(1024, total_train // 2), 
                                          min(1024 + 256, total_train)))
        
        processed_dataset["train"] = train_split
        processed_dataset["validation"] = val_split
    
    if "test" in raw_dataset:
        test_data = raw_dataset["test"].map(format_sample)
        processed_dataset["test"] = test_data.select(range(min(2048, len(test_data))))
    else:
        # If no test split, use remaining data
        remaining_data = train_data.select(range(min(1024 + 256, total_train), total_train))
        processed_dataset["test"] = remaining_data.select(range(min(2048, len(remaining_data))))
    
    # Create dataset object
    from datasets import DatasetDict
    dataset = DatasetDict(processed_dataset)
    
    # Push to Hugging Face Hub
    dataset.push_to_hub("EuroEval/exam_et", private=True)
    print("Dataset successfully processed and uploaded!")


if __name__ == "__main__":
    process_exam_et()
