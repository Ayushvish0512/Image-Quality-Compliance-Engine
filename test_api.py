"""
Quick functional test for the VCE API.
"""
import sys
sys.path.insert(0, '.')

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# 1. Test health
r = client.get('/health')
print('=== Health Endpoint ===')
print('Status:', r.status_code)
print('Response:', r.json())
print()

# 2. Test validation with a real image URL (GitHub logo)
print('=== Validate Endpoint ===')
TEST_URL = 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png'
r = client.post('/validate', json={'image_url': TEST_URL})
print('Status:', r.status_code)

if r.status_code == 200:
    data = r.json()
    print('Overall Score:', data.get('overall_score'))
    print('Status:', data.get('status'))

    checks = data.get('checks', {})
    print('\nChecks:')
    print('  Person count:', checks.get('person_count'))
    print('  Face detected:', checks.get('face_detected'))
    print('  Face visibility:', checks.get('face_visibility'))
    print('  Blur score:', checks.get('blur_score'))
    print('  Lighting:', checks.get('lighting'))
    print('  Distance:', checks.get('distance'))
    print('  Orientation:', checks.get('orientation'))
    print('  Uniform:', checks.get('uniform'))
    print('  Cap:', checks.get('cap'))
    print('  Screenshot risk:', checks.get('screenshot_risk'))

    print('\nRecommendations:', len(data.get('recommendations', [])))
    print('Has score_breakdown:', 'score_breakdown' in data)

    if 'score_breakdown' in data:
        print('\nScore Breakdown:')
        for k, v in data['score_breakdown'].items():
            print(f'  {k}: {v}')
else:
    print('Error:', r.text[:1000])
