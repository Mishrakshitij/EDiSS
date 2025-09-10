# Setting Up the Dataset Request Form

This guide will help you set up the dataset request form to receive and manage access requests.

## Option 1: Using Formspree (Recommended - Free Tier Available)

1. **Sign up at [Formspree.io](https://formspree.io)**
   - Create a free account
   - Free tier allows 50 submissions/month

2. **Create a new form**
   - Click "New Form"
   - Name it "PDCare Dataset Requests"
   - Copy your form endpoint (looks like: `https://formspree.io/f/YOUR_FORM_ID`)

3. **Update the HTML form**
   - Open `index.html`
   - Replace `YOUR_FORM_ID` in the form action with your actual Formspree form ID:
   ```html
   <form id="datasetRequestForm" action="https://formspree.io/f/YOUR_ACTUAL_ID" method="POST">
   ```

4. **Configure Formspree settings**
   - Add your email to receive notifications
   - Enable file uploads (for ID verification)
   - Set up auto-reply message (optional)

## Option 2: Using Google Forms

1. **Create a Google Form**
   - Go to [Google Forms](https://forms.google.com)
   - Create a new form with all the fields from the HTML form
   - Enable file upload for ID verification

2. **Get the embed code**
   - Click "Send" → "Embed HTML"
   - Copy the iframe code

3. **Replace the form**
   - Replace the form in `index.html` with the Google Forms iframe
   - Or link to the Google Form directly

## Option 3: Using Netlify Forms (If hosting on Netlify)

1. **Add Netlify attribute**
   ```html
   <form id="datasetRequestForm" data-netlify="true" name="dataset-request">
   ```

2. **Deploy to Netlify**
   - Push your code to GitHub
   - Connect repository to Netlify
   - Forms will automatically work

## Option 4: Custom Backend Solution

### Using Node.js + Express + Nodemailer

Create `server.js`:

```javascript
const express = require('express');
const multer = require('multer');
const nodemailer = require('nodemailer');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

// Configure nodemailer
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'your-email@gmail.com',
    pass: 'your-app-password'
  }
});

// Handle form submission
app.post('/submit-request', upload.single('idUpload'), async (req, res) => {
  try {
    const formData = req.body;
    const file = req.file;
    
    // Email to admin
    const adminMailOptions = {
      from: 'your-email@gmail.com',
      to: 'mishra.kshitij07@gmail.com',
      subject: 'New PDCare Dataset Request',
      html: `
        <h2>New Dataset Request</h2>
        <p><strong>Name:</strong> ${formData.firstName} ${formData.lastName}</p>
        <p><strong>Email:</strong> ${formData.email}</p>
        <p><strong>Organization:</strong> ${formData.organization}</p>
        <p><strong>Position:</strong> ${formData.position}</p>
        <p><strong>Country:</strong> ${formData.country}</p>
        <p><strong>Research Purpose:</strong> ${formData.researchPurpose}</p>
        <p><strong>Project Title:</strong> ${formData.projectTitle}</p>
        <p><strong>Expected Outcome:</strong> ${formData.expectedOutcome}</p>
        <p><strong>Profile URL:</strong> ${formData.profileUrl}</p>
        <p><strong>ID Document:</strong> Attached</p>
      `,
      attachments: [{
        filename: file.originalname,
        path: file.path
      }]
    };
    
    // Auto-reply to requester
    const userMailOptions = {
      from: 'your-email@gmail.com',
      to: formData.email,
      subject: 'PDCare Dataset Request Received',
      html: `
        <h2>Thank you for your interest in the PDCare Dataset</h2>
        <p>Dear ${formData.firstName} ${formData.lastName},</p>
        <p>We have received your request for access to the PDCare dataset. 
        Your application will be reviewed within 2-3 business days.</p>
        <p>If approved, you will receive the dataset via email along with 
        detailed usage instructions.</p>
        <p>Best regards,<br>EDiSS Team</p>
      `
    };
    
    // Send emails
    await transporter.sendMail(adminMailOptions);
    await transporter.sendMail(userMailOptions);
    
    res.json({ success: true, message: 'Request submitted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: 'Error processing request' });
  }
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
```

### Update HTML form JavaScript

```javascript
document.getElementById('datasetRequestForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    
    try {
        const response = await fetch('http://localhost:3000/submit-request', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('successMessage').style.display = 'block';
            this.reset();
        } else {
            document.getElementById('errorMessage').style.display = 'block';
        }
    } catch (error) {
        document.getElementById('errorMessage').style.display = 'block';
    }
});
```

## Managing Requests

### Create a Tracking Spreadsheet

Create `Dataset/requests_tracker.csv`:

```csv
Date,Name,Email,Organization,Position,Country,Purpose,Status,Notes,Date_Sent
2024-01-15,John Doe,john@university.edu,University X,PhD Student,USA,Thesis research,Approved,Verified,2024-01-16
```

### Review Process Checklist

Create `Dataset/review_checklist.md`:

```markdown
# Dataset Request Review Checklist

For each request, verify:

- [ ] Valid institutional email address
- [ ] Legitimate organization/institution
- [ ] Clear research purpose
- [ ] ID document matches provided information
- [ ] No commercial intent (unless authorized)
- [ ] Profile URL validates identity
- [ ] Research aligns with ethical use

## Approval Email Template

Subject: PDCare Dataset Access Approved

Dear [Name],

Your request for the PDCare dataset has been approved. 

Please find the dataset attached/linked below:
[Secure download link]

Remember to:
- Use the dataset only for the stated research purpose
- Cite our paper in any publications
- Do not redistribute the dataset
- Follow ethical guidelines

Best regards,
EDiSS Team

## Rejection Email Template

Subject: PDCare Dataset Request Update

Dear [Name],

Thank you for your interest in the PDCare dataset. 

After reviewing your application, we need additional information:
[Specific concerns/requirements]

Please feel free to resubmit with the requested information.

Best regards,
EDiSS Team
```

## Security Considerations

1. **Store requests securely** - Use encrypted storage for submitted documents
2. **Use HTTPS** - Ensure the form is served over HTTPS
3. **Validate inputs** - Implement server-side validation
4. **Rate limiting** - Prevent spam submissions
5. **GDPR compliance** - Include privacy policy and data handling information

## Testing the Form

1. Test with different file types and sizes
2. Verify email delivery
3. Check mobile responsiveness
4. Test form validation
5. Ensure accessibility compliance

## Deployment Options

- **GitHub Pages**: Host the static HTML form (with Formspree)
- **Netlify**: Free hosting with built-in form handling
- **Vercel**: Deploy with serverless functions
- **Your Institution's Server**: Use institutional hosting

Remember to update the form action URL in `index.html` with your chosen solution!