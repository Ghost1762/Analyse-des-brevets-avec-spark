const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

mongoose.connect('mongodb://localhost:27017/commentsDB')
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('Could not connect to MongoDB', err));

// Définir le schéma et le modèle de commentaires
const commentSchema = new mongoose.Schema({
  username: String,
  userId: String, // Champ pour stocker l'identifiant de l'utilisateur
  date: { type: Date, default: Date.now },
  comment: String,
  language: String,
  sentiment: String
});

const Comment = mongoose.model('Comment', commentSchema);

// Route pour gérer la soumission du formulaire de commentaires
app.post('/api/comments', async (req, res) => {
  const { username, userId, comment, language, date } = req.body;

  try {
    const response = await axios.post('http://localhost:5000/predict', { comment: comment, language: language });
    const sentiment = response.data.prediction;
    const newComment = new Comment({ username, userId, date, comment, language, sentiment });
    await newComment.save();
    res.status(200).send('Comment saved successfully');
  } catch (err) {
    console.error('Error saving comment', err);
    res.status(500).send('Error saving comment');
  }
});

// Route pour récupérer les commentaires
app.get('/api/comments', async (req, res) => {
  try {
    const comments = await Comment.find().sort({ date: -1 });
    res.json(comments);
  } catch (err) {
    console.error('Error retrieving comments', err);
    res.status(500).send('Error retrieving comments');
  }
});

// Route pour supprimer un commentaire
app.delete('/api/comments/:id', async (req, res) => {
  const commentId = req.params.id;
  const { username } = req.body; // Nom d'utilisateur effectuant la demande

  // Vérifier si l'ID est valide
  if (!mongoose.Types.ObjectId.isValid(commentId)) {
    return res.status(400).send('Invalid ID format');
  }

  try {
    const comment = await Comment.findById(commentId);

    if (!comment) {
      return res.status(404).send('Comment not found');
    }

    // Vérifier que l'utilisateur actuel est bien le propriétaire du commentaire
    if (comment.username !== username) {
      return res.status(403).send('You are not authorized to delete this comment');
    }

    await Comment.findByIdAndDelete(commentId);
    res.status(200).send('Comment deleted successfully');
  } catch (err) {
    console.error('Error deleting comment', err);
    res.status(500).send('Error deleting comment');
  }
});

// Route pour compter les commentaires positifs et négatifs
app.get('/api/comments/count', async (req, res) => {
  try {
    const positiveCount = await Comment.countDocuments({ sentiment: 'positive' });
    const negativeCount = await Comment.countDocuments({ sentiment: 'negative' });
    res.json({ positive: positiveCount, negative: negativeCount });
  } catch (err) {
    console.error('Error counting comments', err);
    res.status(500).send('Error counting comments');
  }
});


// Démarrer le serveur
const port = process.env.PORT || 3001;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
