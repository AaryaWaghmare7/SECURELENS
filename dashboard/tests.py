from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .models import ImageAnalysis


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
