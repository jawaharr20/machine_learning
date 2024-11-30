import numpy as np
from collections import defaultdict

# Sample dataset of emails and their labels (1 for not spam, 0 for spam)
emails = [
    ('Buy now, limited offer!', 0),
    ('Hello, how are you?', 1),
    ('Exclusive deal just for you', 0),
    ('Meeting agenda for next week', 1),
    ('Claim your prize now', 0)
]

# Function to tokenize words from text
def tokenize(text):
    return text.lower().split()

# Initialize dictionaries to store word counts in spam and non-spam emails
word_count_spam = defaultdict(int)
word_count_ham = defaultdict(int)

# Process each email to populate word counts
total_spam = 0
total_ham = 0

for email, label in emails:
    words = tokenize(email)
    if label == 1:
        for word in words:
            word_count_ham[word] += 1
        total_ham += 1
    else:
        for word in words:
            word_count_spam[word] += 1
        total_spam += 1

# Function to calculate the probability of each word being in spam or ham emails with Laplace Smoothing
def calculate_word_probabilities(word_counts, total_count, vocab_size, alpha=1):
    probabilities = {}
    for word, count in word_counts.items():
        probabilities[word] = (count + alpha) / (total_count + alpha * vocab_size)
    return probabilities

# Get the size of the vocabulary (unique words across all emails)
vocab = set(word_count_spam.keys()).union(set(word_count_ham.keys()))
vocab_size = len(vocab)

# Calculate probabilities of each word being spam or ham with smoothing
prob_word_spam = calculate_word_probabilities(word_count_spam, total_spam, vocab_size)
prob_word_ham = calculate_word_probabilities(word_count_ham, total_ham, vocab_size)

# Prior probabilities (probability of an email being spam or ham)
p_spam = sum(1 for _, label in emails if label == 0) / len(emails)
p_ham = 1 - p_spam

# Function to predict if an email is spam using Bayes' Theorem
def predict_spam(email):
    words = tokenize(email)
    log_p_spam_given_email = np.log(p_spam)
    log_p_ham_given_email = np.log(p_ham)
    
    for word in words:
        # Add smoothing to handle unseen words in new emails
        log_p_spam_given_email += np.log(prob_word_spam.get(word, 1 / (total_spam + vocab_size)))
        log_p_ham_given_email += np.log(prob_word_ham.get(word, 1 / (total_ham + vocab_size)))

    # Normalize probabilities using log-sum-exp trick
    max_log_prob = max(log_p_spam_given_email, log_p_ham_given_email)
    p_spam_given_email = np.exp(log_p_spam_given_email - max_log_prob)
    p_ham_given_email = np.exp(log_p_ham_given_email - max_log_prob)
    
    return p_spam_given_email / (p_spam_given_email + p_ham_given_email)

# Test the model with new emails
test_emails = [
    'Limited offer, claim your prize now!',
    'Hello, how about meeting for lunch tomorrow?',
    'Exclusive deal just for you'
]

for email in test_emails:
    spam_probability = predict_spam(email)
    print(f"Email: '{email}'")
    print(f"Spam Probability: {spam_probability:.4f}")
    if spam_probability > 0.5:
        print("Prediction: Spam\n")
    else:
        print("Prediction: Not Spam\n")
