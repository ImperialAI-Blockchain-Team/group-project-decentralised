import React from 'react';
import { fireEvent, getByTestId, getByText, render,screen, waitFor } from '@testing-library/react';


import Enzyme from 'enzyme';
//import Adapter from 'enzyme-adapter-react-16';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17'
Enzyme.configure({ adapter: new Adapter() });

import {shallow} from 'enzyme'
import {UploadModelForm} from '../components/forms/model.js'

import userEvent from "@testing-library/user-event";

/**
 * Test What user would see
 */
 test('renders the form correctly', () => {
    render(<UploadModelForm />);
    const nameLabel = screen.getByText('Model Name')
    const objLabel = screen.getByText('Objective')
    const modelLabel = screen.getByText ('Model:')
    const descriptionLabel = screen.getByText('Description');
    const reqLabel = screen.getByText('Data Requirements');
    
    expect(nameLabel).toBeInTheDocument();
    expect(descriptionLabel).toBeInTheDocument();
    expect(modelLabel).toBeInTheDocument();
    expect(objLabel).toBeInTheDocument();
    expect(reqLabel).toBeInTheDocument();
    
    const registerBtn = screen.getByTestId("submit")
    //const chooseFile = screen.getByTestId("file")
    expect(registerBtn).toBeInTheDocument();
    //expect(chooseFile).toBeInTheDocument();
    
    //execute click
    userEvent.click(registerBtn);
   // userEvent.click(chooseFile);
  })




const simulateChangeOnInput = (wrapper,inputname, inputSelector, newValue) => {
    let input = wrapper.find(inputSelector)
    input.simulate('change',{
        target: {name:inputname,value: newValue},
    })
    return wrapper.find(inputSelector)
}



describe('uploading a file', () => {
    

    
    let file;

    beforeEach(() => {
        file = new File(['(⌐□_□)'], 'chucknorris.py', { type: 'script/py' });
    });
    
    test('file upload',async () =>{

        const { getByTestId } = render(<UploadModelForm />);
        
        // get the upload button
        let uploader = getByTestId('file');

        // simulate ulpoad event and wait until finish 
        await waitFor(() =>
        fireEvent.change(uploader, {
            target: { files: [file] },
        })
        );
        // get the same uploader from the dom    
        let uploadedFile = document.getElementById('fileUpload');
        // check if the file is there
        expect(uploadedFile.files[0].name).toBe('chucknorris.py');
        expect(uploadedFile.files.length).toBe(1);
        
        
        
           
    });
    
    

    it('lets me fill out the form', () => {
        const wrapper = shallow (<UploadModelForm/>)
        const nameInput = simulateChangeOnInput(wrapper,'name','input#name-input','CovidNet')  //second argument is the id of the input field 
        const descriptionInput = simulateChangeOnInput(wrapper,'description','#description-input','Simple model')
        const objectiveInput = simulateChangeOnInput(wrapper,'objective','input#objective-input','abcd') 
        const reqInput = simulateChangeOnInput(wrapper,'dataRequirements','#req-input','abcd') 

        expect(nameInput.props().value).toEqual('CovidNet')
        expect(descriptionInput.props().value).toEqual('Simple model')
        expect(objectiveInput.props().value).toEqual('abcd')
        expect(reqInput.props().value).toEqual('abcd')

        

    });
    

    
})
