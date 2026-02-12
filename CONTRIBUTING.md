# Contributing to Smart Health Tracker Insights

Thank you for your interest in contributing to this project! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions! Please create an issue describing:
- The enhancement you'd like to see
- Why it would be useful
- Any implementation ideas

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed
4. **Test your changes**
   ```bash
   python -m pytest tests/
   ```
5. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/AmazingFeature
   ```
7. **Open a Pull Request**

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Smart-Health-Tracker-Insights.git
cd Smart-Health-Tracker-Insights

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Testing

Before submitting a PR, ensure:
- All existing tests pass
- New features have tests
- Code follows style guidelines

```bash
# Run tests
python -m pytest

# Check code style
flake8 src/
black src/ --check
```

## Questions?

Feel free to open an issue for any questions or clarifications!

---

Thank you for contributing! 🙏
