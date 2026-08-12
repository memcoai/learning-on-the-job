"""Learning on the job: an agent that learns a business's unwritten policies.

The harness runs one loop per task — agent drafts, reviewer corrects, reflection
turns the correction into a lesson held in Memco memory — and records how many
policies the draft breached, so the effect of memory can be measured rather than
asserted.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
