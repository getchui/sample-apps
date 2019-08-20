const express = require('express')
const bodyParser = require('body-parser')
const cors = require('cors')
const axios = require('axios')
const path = require('path')

const app = express()

app.use(express.static(path.join(__dirname, '/build/')))
app.use(bodyParser.json({ limit: '50mb' }))
app.use(cors())
app.enable('trust proxy')

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*")
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
    next()
})

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname + '/build/index.html'))
})

app.post('/predict', (req, res) => {

    axios.post("http://agebox:8080/predict", { "image": req.body.image }, {
        headers: {
            'Content-Type': 'application/json',
        }
    })
        .then(response => {
            res.send(response.data)
        })
        .catch((err) => {
            if (err.response) {
                console.log(err.response.data)
                console.log(err.response.status)
                console.log(err.response.headers)
            } else if (err.request) {
                console.log(err.request)
            } else {
                console.log("Error", error.message)
            }
        })
})

app.listen(4000, ()=> 
            console.log('Server is listening on port 4000')
            )
            
module.exports = app
