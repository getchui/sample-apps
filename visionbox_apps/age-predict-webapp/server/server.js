const express = require('express')
const bodyParser = require('body-parser')
const cors = require('cors')
const path = require('path')
const axios = require('axios')

const app = express()
app.use(express.static(path.join(__dirname, './build')))
app.use(bodyParser.json({ limit: '50mb' }))
app.use(cors())
app.enable('trust proxy')

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, './build', 'index.html'))
})

app.post('/predict', (req, res) => {

    axios.post("http://localhost:9000/predict", { "image": req.body.image }, {
        headers: {
            'Content-Type': 'application/json',
        }
    })
        .then(response => {
            console.log(response.data)
            res.send(response.data)
        })
        .catch(err => console.error(err))
})

app.listen(4000, ()=> 
            console.log('Server is listening on port 4000')
            )
            
module.exports = app