from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .models import ImageAnalysis
from .views import classify_prediction, summarize_model_results


class SecureLensAccessTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='arya', password='strong-pass-123')
        self.other_user = User.objects.create_user(username='friend', password='strong-pass-123')

    def test_register_page_loads(self):
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Create your SecureLens account')

    def test_analyze_requires_login(self):
        response = self.client.get(reverse('analyze'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response.url)

    def test_history_only_shows_current_user_items(self):
        ImageAnalysis.objects.create(owner=self.user, image='uploads/user.jpg', prediction='REAL', confidence=91.2)
        ImageAnalysis.objects.create(owner=self.other_user, image='uploads/other.jpg', prediction='AI', confidence=77.4)

        self.client.login(username='arya', password='strong-pass-123')
        response = self.client.get(reverse('history'))

        self.assertEqual(response.status_code, 200)
        analyses = list(response.context['analyses'])
        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0].owner, self.user)

    def test_stats_page_uses_user_specific_counts(self):
        ImageAnalysis.objects.create(owner=self.user, image='uploads/a.jpg', prediction='REAL', confidence=88.0)
        ImageAnalysis.objects.create(owner=self.user, image='uploads/b.jpg', prediction='AI', confidence=64.0)
        ImageAnalysis.objects.create(owner=self.other_user, image='uploads/c.jpg', prediction='AI', confidence=99.0)

        self.client.login(username='arya', password='strong-pass-123')
        response = self.client.get(reverse('stats'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total'], 2)
        self.assertEqual(response.context['real_count'], 1)
        self.assertEqual(response.context['ai_count'], 1)


class SecureLensPredictionTests(TestCase):
    def test_model_summary_pulls_ai_and_real_scores(self):
        results = [
            {'label': 'AI generated', 'score': 0.81},
            {'label': 'real photograph', 'score': 0.17},
        ]
        summary = summarize_model_results(results)
        self.assertEqual(summary['ai_score'], 0.81)
        self.assertEqual(summary['real_score'], 0.17)

    def test_generic_labels_do_not_force_real(self):
        results = [
            {'label': 'LABEL_0', 'score': 0.92},
            {'label': 'LABEL_1', 'score': 0.08},
        ]
        summary = summarize_model_results(results)
        self.assertEqual(summary['ai_score'], 0.0)
        self.assertEqual(summary['real_score'], 0.0)
        self.assertEqual(summary['unknown_score'], 0.92)

    def test_close_votes_become_uncertain(self):
        prediction, confidence, explanation = classify_prediction(
            [
                {'ai_score': 0.59, 'real_score': 0.55, 'weight': 1.0},
                {'ai_score': 0.54, 'real_score': 0.57, 'weight': 0.9},
            ],
            {'std_pixel': 40.0, 'edge_density': 0.03},
        )
        self.assertEqual(prediction, 'UNCERTAIN')
        self.assertGreater(confidence, 50)
        self.assertIn('too close', explanation.lower())

    def test_clear_majority_returns_real(self):
        prediction, confidence, explanation = classify_prediction(
            [
                {'ai_score': 0.32, 'real_score': 0.78, 'weight': 1.0},
                {'ai_score': 0.28, 'real_score': 0.73, 'weight': 0.9},
                {'ai_score': 0.35, 'real_score': 0.69, 'weight': 0.8},
            ],
            {'std_pixel': 62.0, 'edge_density': 0.09},
        )
        self.assertEqual(prediction, 'REAL')
        self.assertGreater(confidence, 70)
        self.assertIn('real photograph', explanation.lower())
