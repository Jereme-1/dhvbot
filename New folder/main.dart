import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(DHVBotApp());
}

class DHVBotApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DHVBOT AI',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
      ),
      home: ChatScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  List<Map<String, String>> messages = [];
  TextEditingController _controller = TextEditingController();
  bool isLoading = false;

  Future<void> sendMessage(String message) async {
    setState(() {
      messages.add({'role': 'user', 'content': message});
      isLoading = true;
    });

    try {
      final response = await http.post(
        Uri.parse('http://192.168.1.119:8000/ask'), // ← Change if using real phone
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'message': message}),
      );

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);
        setState(() {
          messages.add({'role': 'bot', 'content': responseData['response'] ?? 'No response.'});
        });
      } else {
        setState(() {
          messages.add({'role': 'bot', 'content': 'Error: ${response.statusCode}'});
        });
      }
    } catch (e) {
      setState(() {
        messages.add({'role': 'bot', 'content': '❌ Connection error.'});
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }

    _controller.clear();
  }

  Widget buildMessage(Map<String, String> message) {
    final isUser = message['role'] == 'user';
    return Container(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      padding: EdgeInsets.symmetric(vertical: 6, horizontal: 12),
      child: Container(
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.7),
        padding: EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: isUser ? Colors.deepPurple[100] : Colors.grey[200],
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          message['content'] ?? '',
          style: TextStyle(fontSize: 15),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('🤖 DHVBOT AI'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: EdgeInsets.all(8),
              itemCount: messages.length,
              itemBuilder: (context, index) {
                return buildMessage(messages[index]);
              },
            ),
          ),
          if (isLoading) Padding(
            padding: const EdgeInsets.only(bottom: 8.0),
            child: CircularProgressIndicator(),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Ask something...',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                SizedBox(width: 8),
                ElevatedButton(
                  onPressed: () {
                    final text = _controller.text.trim();
                    if (text.isNotEmpty && !isLoading) {
                      sendMessage(text);
                    }
                  },
                  child: Text('Send'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
