import numpy as np
from collections import defaultdict

# Sample dataset of emails and their labels (1 for spam, 0 for not spam)
emails = [
    ('Buy now, limited offer!', 1),
    ('Hello, how are you?', 0),
    ('Exclusive deal just for you', 1),
    ('Meeting agenda for next week', 0),
    ('Claim your prize now', 1)
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
            word_count_spam[word] += 1
            total_spam += 1
    else:
        for word in words:
            word_count_ham[word] += 1
            total_ham += 1

# Function to calculate probability of each word being spam
def calculate_word_probabilities(word_counts, total_count):
    probabilities = {}
    for word, count in word_counts.items():
        probabilities[word] = count / total_count
    return probabilities

# Calculate probabilities of each word being spam or ham
prob_word_spam = calculate_word_probabilities(word_count_spam, total_spam)
prob_word_ham = calculate_word_probabilities(word_count_ham, total_ham)

# Prior probabilities (probability of an email being spam or ham)
p_spam = sum(1 for _, label in emails if label == 1) / len(emails)
p_ham = 1 - p_spam

# Function to predict if an email is spam using Bayes' Theorem
def predict_spam(email):
    words = tokenize(email)
    log_p_spam_given_email = np.log(p_spam)
    log_p_ham_given_email = np.log(p_ham)
    
    for word in words:
        if word in prob_word_spam:
            log_p_spam_given_email += np.log(prob_word_spam[word])
        if word in prob_word_ham:
            log_p_ham_given_email += np.log(prob_word_ham[word])
    
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
