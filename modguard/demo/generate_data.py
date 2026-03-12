"""Generate diverse sample content for testing the moderation pipeline.

Produces 50+ samples spanning clean, borderline, toxic, and spam categories
to exercise all classifier layers. Can output to stdout or write to a JSON file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def generate_samples() -> list[dict[str, str]]:
    """Generate a list of categorized sample texts.

    Returns:
        List of dictionaries with 'text' and 'category' keys.
    """
    samples: list[dict[str, str]] = []

    # Clean content (should be approved)
    clean = [
        "Great article! I learned a lot about renewable energy.",
        "Thank you for sharing this recipe. My family loved it.",
        "The weather forecast looks promising for the weekend.",
        "I appreciate the detailed explanation of quantum computing.",
        "This library has excellent documentation and examples.",
        "Happy birthday! Wishing you all the best this year.",
        "The new park downtown is beautiful. Perfect for a morning jog.",
        "Just finished reading an amazing book about marine biology.",
        "Our team meeting went really well today. Productive discussion.",
        "Looking forward to the community volunteer event next Saturday.",
        "The sunset over the mountains was breathtaking yesterday.",
        "My garden is finally blooming after weeks of careful watering.",
        "This tutorial on Python decorators was very helpful.",
        "Congratulations on your promotion! Well deserved.",
        "The museum exhibit on ancient civilizations was fascinating.",
    ]
    for text in clean:
        samples.append({"text": text, "category": "clean"})

    # Borderline content (may be flagged for review)
    borderline = [
        "I think that policy is absolutely terrible and whoever supports it is misguided.",
        "This product is a complete waste of money. Total garbage.",
        "People who drive like that should have their licenses revoked permanently.",
        "That movie was painfully bad. I want those two hours of my life back.",
        "Some politicians are so corrupt it makes me sick.",
        "The customer service was unbelievably rude and unhelpful.",
        "I cannot believe how incompetent that referee was. Ruined the whole game.",
        "That restaurant is disgusting. Found something questionable in my food.",
        "People who litter in national parks are the worst kind of people.",
        "This software update completely destroyed my workflow. Whoever approved it should be fired.",
    ]
    for text in borderline:
        samples.append({"text": text, "category": "borderline"})

    # Toxic content (should be rejected)
    toxic = [
        "You are the dumbest person I have ever encountered online.",
        "Shut up, nobody asked for your worthless opinion.",
        "I hate every single person in this thread.",
        "You should be ashamed of yourself for existing.",
        "Everyone who disagrees with me is an absolute moron.",
        "This group is full of brainless idiots who cannot think for themselves.",
        "Go away, loser. Nobody wants you here.",
        "You disgust me. Everything you say is garbage.",
        "What a pathetic excuse for a human being you are.",
        "I hope your project fails spectacularly and everyone laughs at you.",
    ]
    for text in toxic:
        samples.append({"text": text, "category": "toxic"})

    # Spam content (should be flagged or rejected)
    spam = [
        "BUY NOW!!! AMAZING DEAL!!! CLICK HERE!!! LIMITED TIME OFFER!!!",
        "FREE FREE FREE!!! Win a brand new iPhone by clicking this link: http://scam.example.com",
        "CONGRATULATIONS! You have been selected as a WINNER! Act now to claim your prize!",
        "Visit http://spam1.example.com http://spam2.example.com http://spam3.example.com http://spam4.example.com for deals!",
        "MAKE $10000 A DAY FROM HOME!!! NO EXPERIENCE NEEDED!!! CLICK NOW!!!",
        "Subscribe to my channel! Like and share! Follow me everywhere! http://everywhere.example.com",
        "AAAAAAAAAA THIS IS SOOOOO AMAZING BUY NOW!!!!!!!!!!",
        "Free gift cards! Click here now! Limited time! Buy now! Order now! Subscribe!",
        "YOU WON'T BELIEVE THIS SECRET TRICK THAT DOCTORS HATE!!!",
        "ATTENTION: Your account has been compromised. Click here immediately: http://phishing.example.com",
    ]
    for text in spam:
        samples.append({"text": text, "category": "spam"})

    # Sarcastic content (tests sentiment context flags)
    sarcastic = [
        "Oh sure, because that plan worked out SO well last time.",
        "Yeah right, like anyone would believe that excuse.",
        "Wow, so original. Never heard that one before. /s",
        "What a surprise that the cheapest option broke immediately.",
        "Totally not obvious that was going to fail. Who could have predicted it?",
    ]
    for text in sarcastic:
        samples.append({"text": text, "category": "sarcastic"})

    # Backhanded compliments (tests sentiment context flags)
    backhanded = [
        "Not bad for someone who just started programming.",
        "You're actually pretty smart for your age.",
        "No offense but your presentation could use a lot of work.",
        "I'm surprised you managed to finish on time, honestly.",
        "Pretty good for a first attempt, I suppose.",
    ]
    for text in backhanded:
        samples.append({"text": text, "category": "backhanded"})

    return samples


def write_samples(output_path: str | None = None) -> None:
    """Generate samples and write them to a JSON file.

    Args:
        output_path: Path to write the JSON file. If None, writes to
            the default sample_content.json location.
    """
    samples = generate_samples()

    if output_path is None:
        output_path = str(Path(__file__).parent / "sample_content.json")

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} samples, written to {output_path}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    write_samples(output)
