# EasyMARL Documentation Website

This directory contains the React-based documentation website for EasyMARL that gets deployed to GitHub Pages.

## 🌐 Live Documentation

The documentation is automatically deployed to: **https://shreyanmitra.github.io/EasyMARL**

## 🏗️ Local Development

To run the documentation site locally:

```bash
cd docs-website
npm install
npm start
```

This will start the development server at `http://localhost:3000`.

## 📦 Building for Production

```bash
cd docs-website
npm run build
```

The built files will be in the `build/` directory.

## 🚀 Deployment

The documentation is automatically deployed to GitHub Pages when:

1. **Push to main branch** with changes in `docs-website/` folder
2. **Manual trigger** via GitHub Actions workflow

### Deployment Process

1. GitHub Actions workflow builds the React app
2. Generated files are uploaded to GitHub Pages
3. Site becomes available at the GitHub Pages URL

### GitHub Pages Setup

1. Go to Repository Settings → Pages
2. Set Source to "GitHub Actions"
3. The workflow will handle the rest automatically

## 📁 Structure

```
docs-website/
├── public/           # Static files
├── src/
│   ├── components/   # Reusable React components
│   ├── pages/        # Page components
│   ├── App.js        # Main app component
│   └── index.js      # Entry point
├── package.json      # Dependencies and scripts
└── README.md         # This file
```

## 🎨 Features

- **Professional Design**: PyTorch-inspired styling with modern gradients
- **Responsive Layout**: Works perfectly on all devices
- **Interactive Elements**: Code copying, smooth animations
- **SEO Optimized**: Meta tags and social media cards
- **Fast Loading**: Optimized React build

## 🔧 Technologies Used

- **React 18**: Modern React with hooks
- **React Router**: Client-side routing
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Beautiful icons
- **Framer Motion**: Smooth animations

## 📝 Content Management

To update documentation content:

1. Edit pages in `src/pages/`
2. Update components in `src/components/`
3. Push changes to main branch
4. GitHub Actions will automatically deploy

## 🌟 Key Pages

- **Homepage**: Overview and quick start
- **Documentation**: Installation and usage guides
- **Algorithms**: Detailed algorithm comparisons
- **Tutorials**: Step-by-step learning guides
- **Examples**: Ready-to-run code examples
- **API Reference**: Complete API documentation

## 🎯 Future Enhancements

- [ ] Search functionality
- [ ] Interactive algorithm playground
- [ ] Video tutorials integration
- [ ] Multi-language support
- [ ] Dark mode toggle
- [ ] Performance benchmarks
