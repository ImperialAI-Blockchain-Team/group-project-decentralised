import { render, screen } from '@testing-library/react';
import App from '../App';
import {BrowserRouter} from "react-router-dom";

test('render app', () => {
  render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
      );
  // const linkElement = screen.getByText(/learn react/i);
  // expect(linkElement).toBeInTheDocument();
});
