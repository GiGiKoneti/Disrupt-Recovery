"""
Pytest Configuration and Shared Fixtures

Provides reusable test fixtures for all test modules.

Author: SynthDetect Team
"""

import pytest
import numpy as np


@pytest.fixture
def sample_human_text():
    """A human-written text sample for testing."""
    return (
        "The morning light filtered through the dusty blinds, casting long shadows "
        "across the kitchen floor. Martha shuffled to the coffee maker, her slippers "
        "scuffing against the linoleum. She'd been awake since four, worrying about "
        "the letter from the bank. It sat unopened on the counter, a white rectangle "
        "of dread. The coffee maker gurgled and hissed, filling the small apartment "
        "with its familiar aroma. She poured herself a cup, black, no sugar — the "
        "way her father had always taken his. Through the window she could see the "
        "first joggers of the morning, their breath visible in the cold air. She "
        "envied their simple purpose: one foot in front of the other. If only life's "
        "problems could be outrun so easily. The letter waited. Martha reached for "
        "it, then pulled her hand back. Not yet. First, the coffee."
    )


@pytest.fixture
def sample_ai_text():
    """A text mimicking AI-generated style for testing."""
    return (
        "Artificial intelligence has fundamentally transformed the landscape of modern "
        "technology and society. The rapid advancement of machine learning algorithms, "
        "particularly deep neural networks, has enabled unprecedented capabilities in "
        "natural language processing, computer vision, and decision-making systems. "
        "These developments have significant implications for various sectors, including "
        "healthcare, finance, education, and transportation. Furthermore, the integration "
        "of AI systems into everyday applications has raised important questions about "
        "privacy, bias, and accountability. As we continue to develop and deploy these "
        "technologies, it is essential to establish robust ethical frameworks that ensure "
        "responsible innovation. The potential benefits of AI are substantial, but they "
        "must be balanced against the risks of misuse and unintended consequences."
    )


@pytest.fixture
def short_text():
    """A text that's too short for full analysis."""
    return "This is a short text that doesn't meet the minimum word count."


@pytest.fixture
def sample_embedding():
    """A random 256-dim L2-normalised embedding."""
    vec = np.random.randn(256).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec
