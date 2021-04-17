import React from 'react';
import { getByText, render,screen } from '@testing-library/react';

import Enzyme from 'enzyme';
//import Adapter from 'enzyme-adapter-react-16';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17'
Enzyme.configure({ adapter: new Adapter() });

import {shallow} from 'enzyme'
import {RegisterUserForm} from './user.js'

import userEvent from "@testing-library/user-event";

/**
 * Test What user would see
 */
test('renders the form correctly', () => {
  render(<RegisterUserForm />);
  const nameLabel = screen.getByText('Username')
  const emailLabel = screen.getByText('Email Address');
  const addressLabel = screen.getByText('IP Address');
  const typeLabel = screen.getByText('User Type')
  expect(nameLabel).toBeInTheDocument();
  expect(emailLabel).toBeInTheDocument();
  expect(addressLabel).toBeInTheDocument();
  expect(typeLabel).toBeInTheDocument();

  
  
})

test('checking each checkbox', () =>{
    render(<RegisterUserForm/>);

    const cbEl = screen.getByTestId("check1")
    const cbEl2 = screen.getByTestId("check2")
    const cbEl3 = screen.getByTestId("check3")

    expect(cbEl).toBeInTheDocument();
    expect(cbEl2).toBeInTheDocument();
    expect(cbEl3).toBeInTheDocument();

    expect(cbEl).not.toBeChecked();
    expect(cbEl2).not.toBeChecked();
    expect(cbEl3).not.toBeChecked();

    // Execute the click event of the checkbox
    userEvent.click(cbEl);
    userEvent.click(cbEl2);
    userEvent.click(cbEl3);

    expect(cbEl).toBeChecked();
    expect(cbEl2).toBeChecked();
    expect(cbEl3).toBeChecked();

    // Execute the click event again
    userEvent.click(cbEl);
    userEvent.click(cbEl2);
    userEvent.click(cbEl3);

    expect(cbEl).not.toBeChecked();
    expect(cbEl2).not.toBeChecked();
    expect(cbEl3).not.toBeChecked();
  

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
        const addressInput = simulateChangeOnInput (wrapper,'address','input#address-input','127.0.0.1')
        expect(nameInput.props().value).toEqual('roopa')
        expect(emailInput.props().value).toEqual('roopa@test.com')
        expect(addressInput.props().value).toEqual('127.0.0.1')

        const registerBtn = wrapper.find('button[type="button"]')
        registerBtn.simulate('click')

        

    })

    

    
})
