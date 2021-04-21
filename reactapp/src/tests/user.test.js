import React from 'react';
import { getByText, render,screen } from '@testing-library/react';

import Enzyme from 'enzyme';
//import Adapter from 'enzyme-adapter-react-16';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17'
Enzyme.configure({ adapter: new Adapter() });

import {shallow} from 'enzyme'
import {RegisterUserForm} from '../components/forms/user.js'

import userEvent from "@testing-library/user-event";

/**
 * Test What user would see
 */
test('renders the form correctly', () => {
  render(<RegisterUserForm />);
  const nameLabel = screen.getByText('Username')
  const emailLabel = screen.getByText('Email Address');
  const typeLabel = screen.getByText('User Type')
  expect(nameLabel).toBeInTheDocument();
  expect(emailLabel).toBeInTheDocument();
  expect(typeLabel).toBeInTheDocument();

  
  
})

test('checking each checkbox', () =>{
    render(<RegisterUserForm/>);

    const cbEl = screen.getByTestId("check1")
    const cbEl2 = screen.getByTestId("check2")
    

    expect(cbEl).toBeInTheDocument();
    expect(cbEl2).toBeInTheDocument();
    

    expect(cbEl).not.toBeChecked();
    expect(cbEl2).not.toBeChecked();
    

    // Execute the click event of the checkbox
    userEvent.click(cbEl);
    userEvent.click(cbEl2);
    

    expect(cbEl).toBeChecked();
    expect(cbEl2).toBeChecked();
    

    // Execute the click event again
    userEvent.click(cbEl);
    userEvent.click(cbEl2);
    

    expect(cbEl).not.toBeChecked();
    expect(cbEl2).not.toBeChecked();
    
  

})

const simulateChangeOnInput = (wrapper,inputname, inputSelector, newValue) => {
    let input = wrapper.find(inputSelector)
    input.simulate('change',{
        target: {name:inputname,value: newValue},
    })
    return wrapper.find(inputSelector)
}

describe('RegisterUserForm', () => {
    it('lets me fill out the form', () => {
        const wrapper = shallow (<RegisterUserForm/>)
        const nameInput = simulateChangeOnInput(wrapper,'name','input#name-input','roopa')  //second argument is the id of the input field 
        const emailInput = simulateChangeOnInput(wrapper,'email','input#email-input','roopa@test.com')
        
        expect(nameInput.props().value).toEqual('roopa')
        expect(emailInput.props().value).toEqual('roopa@test.com')
       

        const registerBtn = wrapper.find('button[type="button"]')
        registerBtn.simulate('click')

        

    })

    

    
})
