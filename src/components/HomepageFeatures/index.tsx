// src/pages/index.js

import React from 'react';
import { Redirect } from '@docusaurus/router';

export default function Home() {
  // Change '/docs/intro' to the URL of the doc page you want to be the default
  return <Redirect to="/docs/intro" />; 
}