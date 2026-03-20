import time

def display_time(duration):
    """Display the duration in good format"""
    if duration < 60:
        print(f"Trained in {duration:.2f} seconds")
    elif duration < 3600:
        print(f"Trained in {duration / 60:.2f} minutes")
    else:
        print(f"Trained in {duration / 3600:.2f} hours")