from src.models.data_models import Feedback, QAPair, FeedbackRequest
from src.logger import get_logger

logger = get_logger(__name__)


class FeedbackController:
    """Controller for handling feedback operations."""

    @staticmethod
    def add_feedback(qapair: QAPair, feedback_data: FeedbackRequest) -> Feedback:
        """
        Add feedback for a specific EndToEndGeneration.

        :param qapair: QAPair instance
        :param feedback_data: FeedbackRequest instance
        :return: Created Feedback instance
        """
        feedback = Feedback(
            qapair=qapair,
            approve=feedback_data.approve,
            issues=feedback_data.issues,
            comments=feedback_data.comments,
            email=feedback_data.email,
        )
        feedback.save()

        logger.info(f"Feedback added for QAPair {qapair.id}")
        return feedback
