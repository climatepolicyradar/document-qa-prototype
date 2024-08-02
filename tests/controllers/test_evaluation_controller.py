import pytest
from src.controllers.EvaluationController import EvaluationController
from unittest.mock import patch, MagicMock
from catalogue import Registry

@pytest.fixture
def controller():
    """Fixture to create an EvaluationController instance."""
    return EvaluationController()

def test_init(controller):
    """Test that the controller initializes evaluators correctly."""
    assert controller.evaluators is not None

def test_get_all_evaluators(controller):
    """Test retrieval of all evaluators."""
    evals = controller.get_all_evaluators()
    assert isinstance(evals, Registry), "📄 Evaluators should be a Registry"
    
    print(evals.get_entry_points())
    print(evals.get_all())

def test_get_evaluator(controller):
    """Test retrieval of a specific evaluator."""
    eval = controller.get_evaluator('g_eval_policy')
    assert eval is not None, "📄 Evaluator should not be None"
    