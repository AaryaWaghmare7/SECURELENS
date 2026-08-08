from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from dashboard.views import analyze_image_path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class Command(BaseCommand):
    help = 'Evaluate SecureLens on a labeled CIFAKE-style dataset.'

    def add_arguments(self, parser):
        parser.add_argument('--dataset-root', type=str, default='', help='Path containing REAL and FAKE/AI folders.')
        parser.add_argument('--limit', type=int, default=100, help='Maximum images per class.')
        parser.add_argument('--download', action='store_true', help='Download CIFAKE with kagglehub before evaluating.')

    def handle(self, *args, **options):
        dataset_root = options['dataset_root']
        if options['download']:
            try:
                import kagglehub
            except Exception as error:
                raise CommandError(f'kagglehub is not installed or not available: {error}')
            dataset_root = kagglehub.dataset_download('birdy654/cifake-real-and-ai-generated-synthetic-images')

        if not dataset_root:
            raise CommandError('Pass --dataset-root /path/to/dataset or use --download.')

        root = Path(dataset_root).expanduser()
        if not root.exists():
            raise CommandError(f'Dataset path does not exist: {root}')

        samples = self.collect_samples(root, options['limit'])
        if not samples:
            raise CommandError('No labeled images found. Expected folders with names like REAL, real, FAKE, AI, or fake.')

        counts = {
            'total': 0,
            'correct': 0,
            'ai_ai': 0,
            'ai_real': 0,
            'real_ai': 0,
            'real_real': 0,
            'uncertain': 0,
        }

        for image_path, truth in samples:
            result = analyze_image_path(str(image_path), image_name=image_path.name, capture_source='upload')
            prediction = result['prediction']
            counts['total'] += 1
            if prediction == truth:
                counts['correct'] += 1
            if prediction == 'UNCERTAIN':
                counts['uncertain'] += 1
            elif truth == 'AI' and prediction == 'AI':
                counts['ai_ai'] += 1
            elif truth == 'REAL' and prediction == 'AI':
                counts['ai_real'] += 1
            elif truth == 'AI' and prediction == 'REAL':
                counts['real_ai'] += 1
            elif truth == 'REAL' and prediction == 'REAL':
                counts['real_real'] += 1

            self.stdout.write(f'{image_path.name}: truth={truth} prediction={prediction} confidence={result["confidence"]:.2f}%')

        accuracy = counts['correct'] / max(counts['total'], 1)
        precision_ai = counts['ai_ai'] / max(counts['ai_ai'] + counts['ai_real'], 1)
        recall_ai = counts['ai_ai'] / max(counts['ai_ai'] + counts['real_ai'], 1)
        precision_real = counts['real_real'] / max(counts['real_real'] + counts['real_ai'], 1)
        recall_real = counts['real_real'] / max(counts['real_real'] + counts['ai_real'], 1)

        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('SecureLens evaluation complete'))
        self.stdout.write(f'Total: {counts["total"]}')
        self.stdout.write(f'Accuracy: {accuracy:.3f}')
        self.stdout.write(f'AI precision: {precision_ai:.3f}')
        self.stdout.write(f'AI recall: {recall_ai:.3f}')
        self.stdout.write(f'REAL precision: {precision_real:.3f}')
        self.stdout.write(f'REAL recall: {recall_real:.3f}')
        self.stdout.write(f'Uncertain: {counts["uncertain"]}')
        self.stdout.write('Confusion matrix:')
        self.stdout.write(f'  Actual AI   -> Pred AI: {counts["ai_ai"]}, Pred REAL: {counts["real_ai"]}')
        self.stdout.write(f'  Actual REAL -> Pred AI: {counts["ai_real"]}, Pred REAL: {counts["real_real"]}')

    def collect_samples(self, root, limit):
        label_map = {
            'ai': 'AI',
            'fake': 'AI',
            'generated': 'AI',
            'synthetic': 'AI',
            'real': 'REAL',
            'natural': 'REAL',
        }
        samples_by_label = {'AI': [], 'REAL': []}

        for folder in root.rglob('*'):
            if not folder.is_dir():
                continue
            folder_key = folder.name.lower()
            truth = next((label for key, label in label_map.items() if key in folder_key), None)
            if not truth:
                continue
            images = [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
            samples_by_label[truth].extend(images[:limit])

        samples = []
        samples.extend((path, 'AI') for path in samples_by_label['AI'][:limit])
        samples.extend((path, 'REAL') for path in samples_by_label['REAL'][:limit])
        return samples
